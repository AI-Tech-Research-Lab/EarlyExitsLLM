import sys
import os
sys.path.append(os.path.join(sys.path[0], '../../EleutherAI_Eval_harness'))
sys.path.append(os.path.join(sys.path[0], '../../'))

from lm_eval.models.mistral_models import Mistral_7b
from lm_eval.models.mamba_models import Mamba_7b
from lm_eval import evaluate, simple_evaluate
from lm_eval.tasks import get_task_dict
from lm_eval.utils import make_table

# os.chdir(os.path.join(sys.path[0], './EE_Clean'))
from models.mamba.models.wrapper_mamba_EE import Mamba
from models.mamba.mistral_inference.args import MambaArgs
from models.mistral.tokenizer import Tokenizer

import json
import torch

path = "./weights/mamba/mamba-codestral-7B-v0.1/"
max_length = 2048*2
max_gen_tokens = 64
device = "cuda"
batch_size = 1

# create your model (could be running finetuning with some custom modeling code)
print("Loading model...")

with open(path + "params.json", "r") as f:
    model_args = MambaArgs.from_dict(json.load(f))
    print(model_args)

model_args.ee_pos = []
model_args.block_size = 1024*4

model = Mamba(model_args)
# model.to("cuda")

import time
start_time = time.time()
model.from_folder("./weights/mamba/mamba-codestral-7B-v0.1")
print(f"Time to load model: {time.time() - start_time}")

start_time = time.time()
model.to("cuda")
print(f"Time to load model to GPU: {time.time() - start_time}")
n_layer = model_args.n_layers

model.eval()
model.to(device = device, dtype = torch.bfloat16)

print("Loading tokenizer...")
tokenizer = Tokenizer(path + "/tokenizer.model.v3")

# instantiate an LM subclass that takes your initialized model and can run
# - `Your_LM.loglikelihood()`
# - `Your_LM.loglikelihood_rolling()`
# - `Your_LM.generate_until()`
lm_obj = Mamba_7b(model=model, tokenizer = tokenizer, batch_size=batch_size, max_length = max_length, max_gen_tokens = max_gen_tokens,device = device)

# optional: the task_manager indexes tasks including ones
# specified by the user through `include_path`.
# task_manager = lm_eval.tasks.TaskManager(
#     include_path="/path/to/custom/yaml"
#     )

# To get a task dict for `evaluate`
# task_dict = get_task_dict(
#     ["truthfulqa"]
#     )

print("Evaluating...")
# results = evaluate(
#     lm=lm_obj,
#     task_dict=task_dict,
#     # num_fewshot=3,
# )


# for triviaqa in need to generete text
# also for truthfulqa
# hellaswag
results = simple_evaluate(
    model = lm_obj,
    tasks = ["mmlu"],
    num_fewshot = 0,
)

print(make_table(results))