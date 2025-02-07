import sys
import os
sys.path.append(os.path.join(sys.path[0], '../../EleutherAI_Eval_harness'))
sys.path.append(os.path.join(sys.path[0], '../../'))

from lm_eval.models.mamba_models_EE import Mamba_7b
from lm_eval import evaluate, simple_evaluate
from lm_eval.tasks import get_task_dict
from lm_eval.utils import make_table

# os.chdir(os.path.join(sys.path[0], './EE_Clean'))
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from models.mamba.models.wrapper_mamba_EE import Mamba
from models.mamba.mistral_inference.args import MambaArgs
from models.mistral.tokenizer import Tokenizer

import json
import torch

max_length = 2048*2
max_gen_tokens = 64
device = "cuda"
batch_size = 1

# create your model (could be running finetuning with some custom modeling code)
print("Loading model...")

path = f"./weights/mamba"
path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_32_40_48_56"
plot_intermediate_states = True
th_for_EE = 0.35
ee_pos = [int(p) for p in path_weigths_EE.split("_pos_")[-1].split("_")]
# ee_pos = None
recompute_states = True

with open("./weights/mamba/mamba-codestral-7B-v0.1/params.json", "r") as f:
    model_args = MambaArgs.from_dict(json.load(f))
    print(model_args)

model_args.ee_pos = ee_pos
model_args.block_size = max_length

model = Mamba(model_args)
# model.to("cuda")

import time
start_time = time.time()
model.from_folder("./weights/mamba/mamba-codestral-7B-v0.1")
print(f"Time to load model: {time.time() - start_time}")


n_layer = model_args.n_layers

model.th = model.th * th_for_EE if ee_pos is not None else None

start_time = time.time()
model.eval()
model.to(device = device)
print(f"Time to load model to GPU: {time.time() - start_time}")

if ee_pos is not None:
    for i in range(len(ee_pos)):
        model.model.backbone.ee[i].load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))

print("Loading tokenizer...")
# tokenizer = Tokenizer("./weights/mamba/mamba-codestral-7B-v0.1" + "/tokenizer.model.v3")
# tokenizer = MistralTokenizer.v3().instruct_tokenizer
tokenizer = Tokenizer("./weights/mamba/mamba-codestral-7B-v0.1/tokenizer.model.v3")


# instantiate an LM subclass that takes your initialized model and can run
# - `Your_LM.loglikelihood()`
# - `Your_LM.loglikelihood_rolling()`
# - `Your_LM.generate_until()`

for recompute_states in [True, False]:

    lm_obj = Mamba_7b(  model=model,
                        tokenizer = tokenizer,
                        batch_size=batch_size,
                        max_length = max_length,
                        max_gen_tokens = max_gen_tokens,
                        temperature = 1.0,
                        top_k = None,
                        use_EE = True if ee_pos is not None else False,
                        recompute_states = recompute_states,
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


    # triviaqa
    # coqa has to be with n_shots = 0
    # truthfulqa
    results = simple_evaluate(
        model = lm_obj,
        tasks = ["coqa"],
        num_fewshot = 0,
    )

    print(make_table(results))