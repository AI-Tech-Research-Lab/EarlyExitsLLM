from lm_eval.api.model import TemplateLM, CacheHook
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)

import copy
import torch
from torch.functional import F
from tqdm import tqdm

@register_model("mamba7b")
class Mamba_7b(TemplateLM):
    def __init__(self, model, tokenizer, batch_size=4, max_length = 2048, max_gen_tokens = 256, device = "cuda"):
        
        # set rank and world size to a single process, by default.
        self._rank = 0
        self._world_size = 1
        self.cache_hook = CacheHook(None)

        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.logits_cache = True
        self.max_gen_toks = max_gen_tokens
        self.max_length = max_length

        self.device = device


    def tok_encode(self, string: str, **kwargs):
        out = self.tokenizer.encode(string)
        return out
    
    def tok_decode(self, tokens: list[int], **kwargs):
        out = self.tokenizer.decode(tokens)
        return out
    
    def tok_batch_encode(self, strings: list[str], **kwargs):
        out = self.tokenizer._model.encode(strings)
        return out
    
    def tok_batch_decode(self, tokens: list[list[int]], **kwargs):
        out = self.tokenizer._model.decode(tokens)
        return out
    
    def _model_generate(self, context, until):
        with torch.no_grad():
            return self._generate(context, model = self.model, tokenizer = self.tokenizer, until = until, max_tokens = self.max_gen_toks)
    
    def _model_call(self, inputs, **kwargs):
        with torch.no_grad():
            positions = torch.arange(inputs.size(1), device=inputs.device)
            # for the update of july, input in batch size 1, avoiding first dimension
            inputs = inputs[0]

            positions = [len(inputs)]
            return self.model(inputs, positions, **kwargs)

    # def _loglikelihood_tokens_bad(self, requests: list[Instance], disable_tqdm) -> list[tuple[float, bool]]:
        
    #     # inputs, args = requests.args
    #     # max_tokens = args["max_log_toks"]
    #     # res, _ = generate(inputs, self.model, self.tokenizer, max_tokens)

    #     # check number of reqs
    #     n_reqs = len(requests)
    #     n_batches = n_reqs // self.batch_size +1
    #     reqs = [requests[i*self.batch_size:(i+1)*self.batch_size] for i in range(n_batches)]

    #     out = []
    #     for req in reqs:
    #         inputs = [r[0][0] for r in req]
    #         targets = [r[2] for r in req]
            
    #         _, logs = generate(inputs, self.model, self.tokenizer, 1)

    #         out.append(logs[targets])

    #     return out

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> list[tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: tuple[tuple[str, str], list[int], list[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: tuple[tuple[str, str], list[int], list[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts" if self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape
            
                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            batched_inps = pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            ).unsqueeze(0)  # [batch, padding_len_inp]

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs)[0], dim=-1
            ).unsqueeze(0)  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[1] - padding_len_inp)
                )
                logits = logits[:1,ctx_len - contlen : ctx_len]
                # logits = logits.unsqueeze(0)  # [1, contlen, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :]
                    # logits here are [1, contlen, vocab]
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1))
                    # logits = torch.gather(logits.squeeze(0), 1, cont_toks)
                    logits = logits.squeeze(-1)  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)


    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        pass


    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        
        res = []

        def _collate(req: tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            # add EOS token to stop sequences
            eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            
            max_ctx_len = self.max_length - max_gen_toks
            

            # # encode, pad, and truncate contexts for this batch
            # context_enc = self.tok_batch_encode(
            #     list(contexts)
            # )

            # context_enc = context_enc.to(self.device)
            # attn_masks = attn_masks.to(self.device)

            # if "max_length" not in kwargs:
            #     kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # perform batched generation
            if not kwargs["do_sample"]:
                cont_texts = self._model_generate(
                    context=contexts,
                    until=until
                )
            else:
                assert False, "Sampling not supported"

            # cont_toks_list = cont_text.tolist()
            for cont_text, context in zip(cont_texts, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                s = cont_text[len(context):].split("\n\n")[0]

                # s = self.tok_decode(cont_toks)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                # for term in until:
                #     if len(term) > 0:
                #         # ignore '' separator,
                #         # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                #         s = s.split(term)[0]

                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)

        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        return self.eot_token_id
    
    @torch.no_grad()
    def _generate(self, prompts: list[str], model, tokenizer, until, max_tokens: int):
        # encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
        # prompt_lens = [len(x) for x in encoded_prompts]
        # min_prompt_len = min(prompt_lens)
        # max_prompt_len = max(prompt_lens)

        # input_tokens = torch.full((len(prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device="cuda")
        # for i, encoded in enumerate(encoded_prompts):
        #     input_tokens[i, :len(encoded)] = torch.tensor(encoded).to(input_tokens)
        # input_mask = input_tokens != tokenizer.pad_id

        # # pre-fill
        # positions = torch.arange(0, min_prompt_len).to("cuda")
        # logits = model.forward(input_tokens[:, :min_prompt_len], positions)
        # logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        # # decode
        # generated = []
        # all_logprobs = [
        #     logprobs[:,:-1,:].gather(2, input_tokens[:,1:min_prompt_len,None]).squeeze(-1),
        # ]
        # cur_pos = min_prompt_len
        # for _ in range(max_tokens):
        #     next_token = torch.argmax(logprobs[:, -1,:], dim=-1)

        #     if next_token in until:
        #         generated.append(next_token[:, None])

        #         break

        #     if cur_pos < input_mask.shape[1]:
        #         next_token = torch.where(input_mask[:, cur_pos], input_tokens[:, cur_pos], next_token)
        #     all_logprobs.append(
        #         logprobs[:,-1,:].gather(1, next_token[:, None]),
        #     )
        #     generated.append(next_token[:, None])
        #     logits = model.forward(next_token[:, None], torch.LongTensor([cur_pos]).to(next_token))
        #     logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        #     cur_pos += 1

        # all_logprobs = torch.cat(all_logprobs, 1)
        # res = []
        # if max_tokens > 0:
        #     generated = torch.cat(generated, 1)

        #     for i, x in enumerate(encoded_prompts):
        #         res.append(tokenizer.decode(x[:min_prompt_len] + generated[i].tolist()))

               
       
       
       
        return res




   
