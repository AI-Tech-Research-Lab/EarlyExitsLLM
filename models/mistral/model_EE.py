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
from .rope import apply_rotary_emb, apply_rotary_emb_inference, precompute_freqs_cis

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
        
        self.register_buffer("k", torch.zeros((args.block_size, args.n_kv_heads, args.head_dim), device = 'cuda').to(torch.bfloat16))
        self.register_buffer("v", torch.zeros((args.block_size, args.n_kv_heads, args.head_dim), device = 'cuda').to(torch.bfloat16))

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        load_caches = False,
    ) -> torch.Tensor:
        
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # register buffer for k and v caches
        if load_caches:
            self.register_buffer("k", torch.zeros((self.args.block_size, self.n_kv_heads, self.args.head_dim), device = x.device).to(x.dtype))
            self.register_buffer("v", torch.zeros((self.args.block_size, self.n_kv_heads, self.args.head_dim), device = x.device).to(x.dtype))

            self.k[:seqlen_sum, :, :] = xk.view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
            self.v[:seqlen_sum, :, :] = xv.view(seqlen_sum, self.n_kv_heads, self.args.head_dim)

        xq = xq.view(seqlen_sum, self.n_heads, self.args.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        key, val = xk, xv

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...].transpose(1,2), key[None, ...].transpose(1,2), val[None, ...].transpose(1,2)

        # Write the attention function the old fashioned way
        att = (xq @ key.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(key.size(-1))))
        att = att.masked_fill(self.bias[:,:seqlen_sum,:seqlen_sum] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ val # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = y.transpose(1, 2).contiguous().view(seqlen_sum, self.args.dim) # re-assemble all head outputs side by side

        # output projection
        return self.wo(output)
    
    def forward_inference(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        ) -> torch.Tensor:
        
        seqlen_sum, _ = freqs_cis.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(1, self.n_heads, self.args.head_dim)
        self.k[seqlen_sum-1] = xk.view(self.n_kv_heads, self.args.head_dim)
        self.v[seqlen_sum-1] = xv.view(self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb_inference(xq, self.k[:seqlen_sum], freqs_cis=freqs_cis)

        key, val = xk, self.v[:seqlen_sum]

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...].transpose(1,2), key[None, ...].transpose(1,2), val[None, ...].transpose(1,2)

        # Write the attention function the old fashioned way
        att = (xq @ key.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(key.size(-1))))
        att = att.masked_fill(self.bias[:,:seqlen_sum,:seqlen_sum] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att[:,:,-1:] @ val # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = y.transpose(1, 2).contiguous().view(1, 1, self.args.dim) # re-assemble all head outputs side by side

        # output projection
        return self.wo(output.view(1, -1))

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

class EE(nn.Module):
        
        # Linear version of EE
        def __init__(self, args):
            super().__init__()
            self.c_fc = nn.Linear(args.dim, 2, bias = True) #int(args.dim / 2), bias=True)
            # self.c_proj = nn.Linear(int(args.dim / 2), 2, bias=True)
            
        def forward(self, x):
            x = self.c_fc(x)
            # x = self.c_proj(x)
            return x
        
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
        load_caches = False,
    ) -> torch.Tensor:
        r = self.attention(self.attention_norm(x), freqs_cis, load_caches)
        h = x + r

        r = self.feed_forward(self.ffn_norm(h))
        out = h + r

        return out
    
    def forward_inference(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        use_EE = False,
    ) -> torch.Tensor:
        r = self.attention.forward_inference(self.attention_norm(x), freqs_cis)
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

        if args.ee_pos is not None:
            self.ee_pos = args.ee_pos
            self.ee = torch.nn.ModuleList([EE(args) for _ in range(len(args.ee_pos))])

        # set lazily
        self._freqs_cis = None

        self.intermediate_states = torch.zeros(args.n_layers + 1, args.block_size, args.dim, device = 'cuda')

        self.th = torch.ones(args.n_layers - 1, device = 'cuda')

        self.k = 1

        self.recompute_states = False

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
        load_caches = False,
        save_intermediate_states = False,
        train_EE = False,
        n_blocks = 32,
    ) -> torch.Tensor:
        
        seqlens = [input_ids.shape[0]]
        # assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])

        h = self.tok_embeddings(input_ids)
        positions = positions_from_sizes(seqlens, self.freqs_cis.device)
        # att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        if save_intermediate_states:
            self.intermediate_states[0, :sum(seqlens)] = h.detach()

        if train_EE:
            k = self.k
            exits = torch.zeros((1, self.args.block_size, len(self.args.ee_pos), 2), device = input_ids.device)
            early_exits_topk = torch.zeros((1, self.args.block_size, len(self.args.ee_pos), k), device = input_ids.device)
        
        ee_index = 0
        for i,layer in enumerate(self.layers[:n_blocks-1]):

            if train_EE and i != self.args.n_layers - 1:
                h = layer(h, freqs_cis, load_caches)
                if i in self.ee_pos:
                    ee = self.ee[ee_index](h)
                    shape_ee = ee.shape
                    exits[0, :shape_ee[0], ee_index] = ee
                    # early_exits_topk[0, :shape_ee[0], i] = torch.topk(self.lm_head(self.transformer.ln_f(h.detach())), k, dim = 2)[1]
                    early_exits_topk[0, :shape_ee[0], ee_index] = torch.topk(self.output(self.norm(h.detach())).float(), k, dim = 1)[1]
                    ee_index += 1
            else:
                h = layer(h, freqs_cis, load_caches)

            if save_intermediate_states:
                self.intermediate_states[i+1, :sum(seqlens)] = h.detach()

        h = self.layers[-1](h, freqs_cis, load_caches)
        
        logits = self.output(self.norm(h)).float()

        if train_EE:
            
            targets_EE = torch.argmax(logits.detach(), dim = 1).view(1, shape_ee[0], 1).repeat(1,1, len(self.args.ee_pos))
            
            # diff = (targets_EE == torch.argmax(early_exits_topk[:, :shape_ee[1], :], dim = 3)).to(torch.long)

            diff = (targets_EE.unsqueeze(3).repeat(1,1,1,k) == early_exits_topk[:, :shape_ee[0], :]).any(3).to(torch.long)

            ratio = torch.sum(diff)/torch.prod(torch.tensor(diff.shape))

            # ratios = [torch.sum(diff[:,:,i])/torch.prod(torch.tensor(diff[:,:,i].shape)) for i in range(self.args.n_layers)]

            # single loss
            # loss = F.cross_entropy(
            #                         exits[:shape_ee[0],:shape_ee[1]].view(shape_ee[0] * shape_ee[1] * self.config.n_layer, 2),
            #                         diff.view(-1)
            #                         )
            
            
            # if loss is torch.nan:
            #     print("sadge")
            
            
            # sum of losses
            weighted_loss = torch.ones(len(self.args.ee_pos))/len(self.args.ee_pos)
            loss = 0
            losses = torch.empty(len(self.ee_pos))
            ee_index = 0
            for i in range(len(self.args.ee_pos)):
                loss_p = weighted_loss[i] * torch.nn.functional.cross_entropy(
                                    exits[0, :shape_ee[0], ee_index].view(shape_ee[0], 2),
                                    diff[0, :, ee_index].view(-1)
                                    )
                losses[i] = loss_p
                # loss += loss_p
                ee_index += 1

            # print(losses)
            
            # clas weighted single loss
            # loss = F.cross_entropy(
            #                     exits[:shape_ee[0],:shape_ee[1]].view(shape_ee[0] * shape_ee[1] * self.config.n_layer, 2),
            #                     diff.view(-1),
            #                     weight = torch.tensor([ratio, 1 - ratio], device = device, dtype = torch.float)
            #                     )
            

            # compute accuracy between diff and exits
            acc = (diff == torch.argmax(exits[0,:shape_ee[0]], dim = 2)).to(torch.float).mean()
            
            predicted_labels = torch.argmax(exits[0, :shape_ee[0]], dim = 2)

            true_labels = diff.view(-1)
            predicted_labels = predicted_labels.view(-1)

            # True Positives
            TP = ((true_labels == 1) & (predicted_labels == 1)).sum().item()

            # False Positives
            FP = ((true_labels == 0) & (predicted_labels == 1)).sum().item()

            # False Negatives
            FN = ((true_labels == 1) & (predicted_labels == 0)).sum().item()


            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            f1 = 2 * (precision * recall) / (precision + recall)

            metrics = torch.tensor([acc.item(), recall, precision, f1, ratio])

            return logits, losses, metrics

        else:
                   
            return logits, None, None
    
    def forward_inference(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        use_EE = False,
        n_blocks = 32,
    ) -> torch.Tensor:
        # assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])

        h = self.tok_embeddings(input_ids)
        positions = positions_from_sizes(seqlens, self.freqs_cis.device)
        # att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        self.intermediate_states[0, sum(seqlens) - 1] = h.detach()

        ee_index = 0
        for i, layer in enumerate(self.layers[:n_blocks-1]):

            h = layer.forward_inference(h, freqs_cis)

            if use_EE and i in self.args.ee_pos:

                ee = self.ee[ee_index](h)

                ee_sm = torch.nn.functional.softmax(ee[0], dim = 0)
                ind = ee_sm[1] > self.th[ee_index]

                if ind and i < self.args.n_layers - 1:

                    # print("Early exit at layer:", i + 1, " position:", sum(seqlens))
                    
                    if not hasattr(self, 'exits_done'):
                        self.exits_done = []
                    if not hasattr(self, 'positions_exit'):
                        self.positions_exit = []

                    self.exits_done.append(i + 1)
                    self.positions_exit.append(sum(seqlens))

                    # propagate intermediate states
                    for j in range(i+1, self.args.n_layers):

                        if self.recompute_states:
                            norm_x = self.layers[j].attention_norm(h)
                            k = self.layers[j].attention.wk(norm_x)
                            v = self.layers[j].attention.wv(norm_x)

                            # self.transformer.h[j].attn.k[:,pos] = self.transformer.h[i].attn.k[:,pos]
                            # self.transformer.h[j].attn.v[:,pos] = self.transformer.h[i].attn.v[:,pos]

                            self.layers[j].attention.k[sum(seqlens) - 1] = k.view(self.args.n_kv_heads, self.args.head_dim)
                            self.layers[j].attention.v[sum(seqlens) - 1] = v.view(self.args.n_kv_heads, self.args.head_dim)
                        else:
                            self.layers[j].attention.k[sum(seqlens) - 1] = self.layers[j].attention.k[sum(seqlens) - 2]
                            self.layers[j].attention.v[sum(seqlens) - 1] = self.layers[j].attention.v[sum(seqlens) - 2]
                        # self.intermediate_states[0, j+1, pos - 1] = x.detach()
                        # self.intermediate_states[0, j+1, pos - 1] = x.detach()
                        
                    # h = self.layers[-1].forward_inference(h, freqs_cis, use_EE = False)                        

                    break
                    
                ee_index += 1
                
            self.intermediate_states[i+1, sum(seqlens) - 1] = h.detach()

        # if n_blocks != 32:
        h = self.layers[-1].forward_inference(h, freqs_cis, use_EE = False)                        

        return self.output(self.norm(h)).float()
    
    def refresh_caches(self):
        for layer in self.layers:
            del layer.attention.k
            del layer.attention.v

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_EE = False, until = None, recompute_states = False, repetition_penalty = 0, n_blocks = 32):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        self.recompute_states = recompute_states

        # refresh kv caches if exist
        self.refresh_caches()

        # COLD start
        pos_prompt = idx.shape[0]
        logits, _, _ = self(idx, load_caches = True) # just to make sure the model is in the right shape

        logits = logits[-1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float('Inf')

            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim = -1).view(1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next))

        pos = pos_prompt + 1
        break_loop = 0

        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits = self.forward_inference(idx_next, seqlens=[idx.shape[0]], use_EE = use_EE, n_blocks = n_blocks)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[-1, :]
            
            # apply repetition penalty
            if repetition_penalty != 0:
                apply_pen_tokens = torch.where(idx > 894)[0]
                len_mask = len(apply_pen_tokens)
                mask = torch.flip(torch.exp(-torch.arange(len_mask, device = "cuda", dtype = logits.dtype)), dims = (0,)) / (repetition_penalty**2)
                logits[idx[apply_pen_tokens]] = logits[idx[apply_pen_tokens]] * mask

                # we will just square the penalty in the mask
                
                # where = torch.zeros(logits.shape, device = "cuda", dtype=torch.bool)
                # where[idx[apply_pen_tokens]] = True
                # logits = torch.where(where, logits, logits * repetition_penalty) 
            
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[[-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim = -1).view(1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next))

            if until is not None:
                for stops in until:
                    if torch.equal(idx[-len(stops):], torch.tensor(stops, device = self.device)):
                        break_loop = 1
                if break_loop == 1:
                    break
                
        if not hasattr(self, 'lens_generated'):
            self.lens_generated = []
        if break_loop == 1:
            self.lens_generated.append(idx.shape[0] - pos_prompt)
        else:
            self.lens_generated.append(-1)
        
        return idx


    @torch.no_grad()
    def from_pretrained(self, path, dtype: torch.dtype = torch.bfloat16):
    
        sd_hf = load_file(path, device = 'cpu')

        for k, v in sd_hf.items():
            if v.dtype != dtype:
                sd_hf[k] = v.to(dtype)

        self.load_state_dict(sd_hf, strict = False)

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if 'ee' not in name or 'feed' in name:
                param.requires_grad = False

def positions_from_sizes(sizes: Iterable[int], device):
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )

