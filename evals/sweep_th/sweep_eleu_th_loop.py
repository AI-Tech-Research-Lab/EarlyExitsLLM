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
recompute_states = True

# create your model (could be running finetuning with some custom modeling code)
if path_weights is not None:
    with open(path_weights + "/params.json", "r") as f:
        args = ModelArgs.from_dict(json.load(f))

args.max_batch_size = batch_size

print("Loading model...")

path = f"./weights/mistral"
# path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_16_20_24_28"
path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_15_19_23_27"
plot_intermediate_states = True
th_for_EE = 0.7
ee_pos = [int(p) for p in path_weigths_EE.split("_pos_")[-1].split("_")]
# ee_pos = None


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

# lm_obj = Mistral_7b(model=model,
#                     tokenizer = tokenizer,
#                     batch_size=batch_size,
#                     max_length = max_length,
#                     max_gen_tokens = max_gen_tokens,
#                     temperature = 1.0,
#                     top_k = None,
#                     recompute_states = False,
#                     use_EE = True if ee_pos is not None else False,
#                     device = device)

# optional: the task_manager indexes tasks including ones
# specified by the user through `include_path`.
# task_manager = lm_eval.tasks.TaskManager(
#     include_path="/path/to/custom/yaml"
#     )

# To get a task dict for `evaluate`
# task_dict = get_task_dict(
#     ["truthfulqa"]
#     )

# range_th = torch.arange(0, 1.0, 0.1)
range_th = torch.tensor([0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.45, 0.5, 0.7, 1])
# range_th = torch.tensor([1, 0.7])

datasets = ["triviaqa", "coqa", "truthfulqa_gen"]
n_shots = [2, 0, 1]
for dataset,fewshots_set in zip(datasets, n_shots):

    results_list = []
    exits_done = []
    positions_exited = []
    lens_generated = []

    for th in range_th:

        model.th = torch.ones(n_layer - 1) * th

        lm_obj = Mistral_7b(model=model,
            tokenizer = tokenizer,
            batch_size=batch_size,
            max_length = max_length,
            max_gen_tokens = max_gen_tokens,
            temperature = 1.0,
            top_k = None,
            recompute_states = recompute_states,
            use_EE = True,
            device = device)

        # for triviaqa in need to generete text
        # also for truthfulqa
        # nq_open

        # triviaqa WITH n fewshots 2!!!!
        # coqa ONLY posible with n_shots = 0
        # truthfulqa_gen n_shots = 1
        # dataset = "truthfulqa_gen"

        results = simple_evaluate(
            model = lm_obj,
            tasks = [dataset],
            num_fewshot = fewshots_set,
        )

        results_list.append(results)

        if th == 1:
            exits_done.append(0)
            positions_exited.append(None)
        else:
            exits_done.append(lm_obj.model.exits_done)
            lm_obj.model.exits_done = []
            positions_exited.append(lm_obj.model.positions_exit)
            lm_obj.model.positions_exit = []
        lens_generated.append(lm_obj.model.lens_generated)
        lm_obj.model.lens_generated = []

    print(make_table(results_list[0]))

    # save results list, the exits done and the positions, if it does not exist, create it
    if not os.path.exists(path_weigths_EE + f"/results/" + dataset + ("/recompute_states" if recompute_states else "/no_recomp")):
        os.makedirs(path_weigths_EE + f"/results/" + dataset + ("/recompute_states" if recompute_states else "/no_recomp"))

    with open(path_weigths_EE + f"/results/"+dataset+ ("/recompute_states" if recompute_states else "/no_recomp") +"/results_list.json", "w") as f:
        json.dump(results_list, f)

    with open(path_weigths_EE + f"/results/"+dataset+ ("/recompute_states" if recompute_states else "/no_recomp")+ "/exits_done.json", "w") as f:
        json.dump(exits_done, f)

    with open(path_weigths_EE + f"/results/"+dataset+ ("/recompute_states" if recompute_states else "/no_recomp")+"/positions_exited.json", "w") as f:
        json.dump(positions_exited, f)

    with open(path_weigths_EE + f"/results/"+dataset+ ("/recompute_states" if recompute_states else "/no_recomp")+"/lens_generated.json", "w") as f: 
        json.dump(lens_generated, f)

    # save also the th swept
    with open(path_weigths_EE + f"/results/"+dataset+ ("/recompute_states" if recompute_states else "/no_recomp")+"/th_swept.json", "w") as f:
        json.dump(range_th.tolist(), f)
