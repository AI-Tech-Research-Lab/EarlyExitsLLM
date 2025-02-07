import operator
from functools import partial, reduce
from typing import Iterable, List, Optional, Union

import torch
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as torch_ckpt
import torch.nn as nn

from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import AttentionBias, BlockDiagonalCausalMask

from .args import ModelArgs
from .lora import LoRALinear
from .moe import MoeLayer
from .rope import apply_rotary_emb, precompute_freqs_cis, apply_rotary_emb_batch

from safetensors.torch import load_file


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def maybe_lora_layer(
    args: ModelArgs, rank: Optional[int] = None
) -> Union[partial[LoRALinear], type[nn.Linear]]:
    MaybeLora: Union[partial[LoRALinear], type[nn.Linear]]
    if not args.lora.enable:
        return nn.Linear

    rank = rank or args.lora.rank
    scaling = args.lora.scaling
    dropout = args.lora.dropout

    MaybeLora = partial(
        LoRALinear,
        rank=rank,
        scaling=scaling,
        dropout=dropout,
    )

    return MaybeLora


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads
        self.head_dim: int = args.head_dim

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        MaybeLora = maybe_lora_layer(args)

        self.wq = MaybeLora(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = MaybeLora(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = MaybeLora(args.dim, args.n_kv_heads * args.head_dim, bias=False)

        self.wo = MaybeLora(args.n_heads * args.head_dim, args.dim, bias=False)

        self.register_buffer("bias", torch.tril(torch.ones(args.block_size, args.block_size))
                                        .view(1, args.block_size, args.block_size), persistent = False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        
        b, seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(b, seqlen_sum, self.n_heads, self.args.head_dim)
        xk = xk.view(b, seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(b, seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb_batch(xq, xk, freqs_cis=freqs_cis)

        key, val = xk, xv

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=2)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq.transpose(1,2), key.transpose(1,2), val.transpose(1,2)

        # Write the attention function the old fashioned way
        att = (xq @ key.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(key.size(-1))))
        att = att.masked_fill(self.bias[:,:seqlen_sum,:seqlen_sum] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ val # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = y.transpose(1, 2).contiguous().view(b, seqlen_sum, self.args.dim) # re-assemble all head outputs side by side

        # output projection
        return self.wo(output.view(b, seqlen_sum, -1))


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        MaybeLora = maybe_lora_layer(args)
        self.w1 = MaybeLora(args.dim, args.hidden_dim, bias=False)
        self.w2 = MaybeLora(args.hidden_dim, args.dim, bias=False)
        self.w3 = MaybeLora(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)

        self.feed_forward: Union[MoeLayer, FeedForward]
        if args.moe is not None:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],
                gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),
                moe_args=args.moe,
            )
        else:
            self.feed_forward = FeedForward(args=args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        r = self.attention(self.attention_norm(x), freqs_cis)
        h = x + r

        r = self.feed_forward(self.ffn_norm(h))
        out = h + r

        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, checkpoint: bool = False):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(args.n_layers):
            block: torch.nn.Module = TransformerBlock(args=args)
            if checkpoint:
                # activate gradient checkpointing as, see: https://pytorch.org/docs/stable/checkpoint.html
                non_reentrant_wrapper = partial(
                    torch_ckpt.checkpoint_wrapper,
                    checkpoint_impl=torch_ckpt.CheckpointImpl.NO_REENTRANT,
                )
                block = non_reentrant_wrapper(block)

            self.layers.append(block)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = torch.nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )

        self.intermediate_states = torch.zeros((args.n_layers+1, args.block_size, args.dim), device='cuda')

        # set lazily
        self._freqs_cis = None

    @property
    def dtype(self) -> torch.dtype:
        return self.tok_embeddings.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.tok_embeddings.weight.device

    @property
    def freqs_cis(self):
        # lazy init
        device = next(iter(self.parameters())).device
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta=self.args.rope_theta, device=device
            )

        return self._freqs_cis

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        n_blocks = 32,
    ) -> torch.Tensor:
        # assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])

        h = self.tok_embeddings(input_ids)
        positions = positions_from_sizes(seqlens, self.freqs_cis.device)
        # att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)

        self.intermediate_states[0,:h.shape[1]] = h[0]

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        for i,layer in enumerate(self.layers[n_blocks-1:]):
            h = layer(h, freqs_cis)
            self.intermediate_states[i+1,:h.shape[1]] = h[0]
        
        h = self.layers[-1](h, freqs_cis)
        self.intermediate_states[-1,:h.shape[1]] = h[0]

        return self.output(self.norm(h)).float()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits = self(idx, seqlens=[idx.shape[0]])
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:,-1]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim = 1)

        return idx


    @torch.no_grad()
    def from_pretrained(self, path, dtype: torch.dtype = torch.bfloat16):
    
        sd_hf = load_file(path, device = 'cpu')

        for k, v in sd_hf.items():
            sd_hf[k] = v.to(dtype)

        self.load_state_dict(sd_hf)

def positions_from_sizes(sizes: Iterable[int], device):
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )

