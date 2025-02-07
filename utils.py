import torch

from models.gpt2.model_EE import GPT
from models.gpt2.model import GPTConfig

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.protocol.instruct.messages import UserMessage
from models.mistral.model_EE import Transformer, ModelArgs

from models.mamba.models.wrapper_mamba_EE import Mamba
from models.mamba.mistral_inference.args import MambaArgs

import json
import tiktoken

import time


def get_model_and_tok_with_EE(model_choice, path, path_weigths_EE, ee_pos, th_for_EE):
    
    if model_choice == "gpt2":
        
        # open config file
        with open(path + "/config.json") as f:
            config = json.load(f)

            # dump config into GPTConfig
            config_dataclass = GPTConfig(   block_size = config['n_ctx'],
                                            vocab_size = config['vocab_size'],
                                            n_layer = config['n_layer'],
                                            n_head = config['n_head'],
                                            n_embd = config['n_embd'],
                                            dropout = config['attn_pdrop'],
                                            bias = True,
                                        )

            model = GPT(config_dataclass)
            model.from_hf(path + "/model.safetensors")

            model.to("cuda")
            n_layer = model.config.n_layer
            
    elif model_choice == "mistral":
        path = "./weights/mistral/7b-v0.3"
        with open(path+ "/params.json") as f:
            args = ModelArgs(**dict(json.load(f)))
            args.lora.enable = False
            args.ee_pos = ee_pos
            start_time = time.time()
            model = Transformer(args)
            print(f"Time to load model: {time.time() - start_time}")
        start_time = time.time()
        model.from_pretrained(path + "/consolidated.safetensors")
        print(f"Time to load weigths: {time.time() - start_time}")
        start_time = time.time()
        model.to(torch.bfloat16).to("cuda")
        print(f"Time to load model to GPU: {time.time() - start_time}")
        n_layer = model.args.n_layers

    elif model_choice == "mamba":
        path = "./weights/mamba/mamba-codestral-7B-v0.1/"


        with open(path + "params.json", "r") as f:
            model_args = MambaArgs.from_dict(json.load(f))
            print(model_args)

        model_args.ee_pos = ee_pos
        model_args.block_size = 1024*4

        model = Mamba(model_args)
        # model.to("cuda")

        start_time = time.time()
        model.from_folder("./weights/mamba/mamba-codestral-7B-v0.1")
        print(f"Time to load model: {time.time() - start_time}")

        start_time = time.time()
        model.to("cuda")
        print(f"Time to load model to GPU: {time.time() - start_time}")
        n_layer = model_args.n_layers

    if model_choice == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: torch.tensor(enc.encode(s, allowed_special={"<|endoftext|>"}))[None, :]
        decode = lambda l: enc.decode(l[0,:].tolist())
    if model_choice == "mistral" or model_choice == "mamba":
        enc = MistralTokenizer.v3().instruct_tokenizer
        createMsg = lambda s: InstructRequest(messages = [UserMessage(role = "user", content = s)])
        encode = lambda s: torch.tensor(enc.encode_instruct(createMsg(s)).tokens, device = "cuda")
        decode = lambda l: enc.decode(l.tolist())


    model.th = model.th * th_for_EE if ee_pos is not None else None

    if model_choice == "gpt2":
        for i in range(n_layer - 1):
            model.transformer.h[i].ee.load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))
    elif model_choice == "mistral":
        for i in range(len(ee_pos)):
            model.ee[i].load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE", weights_only = False))
    elif model_choice == "mamba":
        for i in range(len(ee_pos)):
            model.model.backbone.ee[i].load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE", weights_only = False))


    return model, encode, decode


def get_model_and_tok_train(model_choice, path, ee_pos, device = "cuda"):

    if model_choice == "gpt2":
        
        # open config file
        with open(path + "/config.json") as f:
            config = json.load(f)

        # dump config into GPTConfig
        config_dataclass = GPTConfig(   block_size = config['n_ctx'],
                                        vocab_size = config['vocab_size'],
                                        n_layer = config['n_layer'],
                                        n_head = config['n_head'],
                                        n_embd = config['n_embd'],
                                        dropout = config['attn_pdrop'],
                                        bias = True,
                                    )

        model = GPT(config_dataclass)
        model.from_hf(path + "/model.safetensors")

        model.to("cuda")
        n_layer = model.config.n_layer
        
    elif model_choice == "mistral":
        path = "./weights/mistral/7b-v0.3"
        with open(path+ "/params.json") as f:
            args = ModelArgs(**dict(json.load(f)))
            args.lora.enable = False
            args.ee_pos = ee_pos
            print("Loading model...")
            model = Transformer(args).to(torch.bfloat16).to("cuda")
        print("Loading weights...")
        model.from_pretrained(path + "/consolidated.safetensors")
        n_layer = model.args.n_layers

    elif model_choice == "mamba":
        path = "./weights/mamba/mamba-codestral-7B-v0.1/"


        with open(path + "params.json", "r") as f:
            model_args = MambaArgs.from_dict(json.load(f))
            print(model_args)

        model_args.ee_pos = ee_pos
        model_args.block_size = 1024*4

        model = Mamba(model_args)
        # model.to("cuda")

        import time
        start_time = time.time()
        model.from_folder("./weights/mamba/mamba-codestral-7B-v0.1")
        print(f"Time to load model: {time.time() - start_time}")

        if device == "cuda":
            start_time = time.time()
            model.to("cuda")
            print(f"Time to load model to GPU: {time.time() - start_time}")

        n_layer = model.args.n_layers
        

    if model_choice == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: torch.tensor(enc.encode(s, allowed_special={"<|endoftext|>"}), device = device)[None, :]
        decode = lambda l: enc.decode(l[0,:].tolist())
    if model_choice == "mistral" or model_choice == "mamba":
        enc = MistralTokenizer.v3().instruct_tokenizer
        createMsg = lambda s: InstructRequest(messages = [UserMessage(role = "user", content = s)])
        encode = lambda s: torch.tensor(enc.encode_instruct(createMsg(s)).tokens, device = device)
        decode = lambda l: enc.decode(l.tolist())

    return model, encode, decode