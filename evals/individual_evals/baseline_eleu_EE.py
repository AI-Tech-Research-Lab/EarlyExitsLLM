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
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


import json
import torch

path_weights = "./weights/mistral/7B-v0.3"
max_length = 2048*2
max_gen_tokens = 32
device = "cuda"
batch_size = 1

# create your model (could be running finetuning with some custom modeling code)
if path_weights is not None:
    with open(path_weights + "/params.json", "r") as f:
        args = ModelArgs.from_dict(json.load(f))

args.max_batch_size = batch_size

print("Loading model...")

path = f"./weights/mistral"
path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_16_20_24_28"
plot_intermediate_states = True
th_for_EE = 0.6
ee_pos = [int(p) for p in path_weigths_EE.split("_pos_")[-1].split("_")]
ee_pos = None


path = "./weights/mistral/7b-v0.3"
with open(path+ "/params.json") as f:
    args = ModelArgs(**dict(json.load(f)))
    args.lora.enable = False
    args.ee_pos = ee_pos
    model = Transformer(args)

print("Loading weights...")
model.from_pretrained(path + "/consolidated.safetensors")
n_layer = model.args.n_layers

model.th = model.th * th_for_EE

if ee_pos is not None:
    for i in range(len(ee_pos)):
        model.ee[i].load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))

print("Sending to GPU...")
model.eval()
if device == "cuda":
    model.to(device = device, dtype = torch.bfloat16)

print("Loading tokenizer...")
tokenizer = Tokenizer(path_weights + "/tokenizer.model.v3")
# tokenizer = MistralTokenizer.v3().instruct_tokenizer

# instantiate an LM subclass that takes your initialized model and can run
# - `Your_LM.loglikelihood()`
# - `Your_LM.loglikelihood_rolling()`
# - `Your_LM.generate_until()`

# model.layers = torch.nn.ModuleList([*model.layers[:27], model.layers[-1]])
drop_layers = 0

lm_obj = Mistral_7b(model=model,
                    tokenizer = tokenizer,
                    batch_size=batch_size,
                    max_length = max_length,
                    max_gen_tokens = max_gen_tokens,
                    temperature = 1.0,
                    top_k = None,
                    recompute_states = True,
                    use_EE = True if ee_pos is not None else False,
                    n_blocks = n_layer - drop_layers,
                    device = device)

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
# nq_open

# triviaqa
# coqa
# generate_until

results = simple_evaluate(
    model = lm_obj,
    tasks = ["truthfulqa_gen"],
    num_fewshot = 1,
)

print(make_table(results))