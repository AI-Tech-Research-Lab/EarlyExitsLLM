from lm_eval.__main__ import cli_evaluate
from argparse import Namespace

args = {"model": "hf",
        "model_args" : {"pretrained" : "gpt2"},
        "tasks": "hellaswag",
        "device": "cuda:0",
        "batch_size": 16}

ns = Namespace(**args)

cli_evaluate(ns)

