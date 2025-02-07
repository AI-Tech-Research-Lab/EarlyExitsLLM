import sys
import os
sys.path.append(os.path.join(sys.path[0], '../../EleutherAI_Eval_harness'))
sys.path.append(os.path.join(sys.path[0], '../../'))

from lm_eval.models.mistral_models_EE import Mistral_7b
from lm_eval import evaluate, simple_evaluate
from lm_eval.tasks import get_task_dict
from lm_eval.utils import make_table

# os.chdir(os.path.join(sys.path[0], './EE_Clean'))

from models.mistral.model import ModelArgs
from models.mistral.model_EE import Transformer
from models.mistral.tokenizer import Tokenizer

import json
import torch

path_weights = "./weights/mistral/7B-v0.3"
max_length = 2048*2
max_gen_tokens = 64
device = "cuda"
batch_size = 1

# create your model (could be running finetuning with some custom modeling code)
if path_weights is not None:
    with open(path_weights + "/params.json", "r") as f:
        args = ModelArgs.from_dict(json.load(f))

args.max_batch_size = batch_size

print("Loading model...")

path = "./weights/mistral/7b-v0.3"
with open(path+ "/params.json") as f:
    args = ModelArgs(**dict(json.load(f)))
    args.lora.enable = False
    model = Transformer(args).to(torch.bfloat16).to("cuda")
model.from_pretrained(path + "/consolidated.safetensors")
n_layer = model.args.n_layers

model.eval()
model.to(device = device, dtype = torch.bfloat16)

print("Loading tokenizer...")
tokenizer = Tokenizer(path_weights + "/tokenizer.model.v3")

# instantiate an LM subclass that takes your initialized model and can run
# - `Your_LM.loglikelihood()`
# - `Your_LM.loglikelihood_rolling()`
# - `Your_LM.generate_until()`
lm_obj = Mistral_7b(model=model, tokenizer = tokenizer, batch_size=batch_size, max_length = max_length, max_gen_tokens = max_gen_tokens,device = device)

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

results = simple_evaluate(
    model = lm_obj,
    tasks = ["hellaswag"],
    num_fewshot = 0,
)

print(make_table(results))