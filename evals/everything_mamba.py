import sys
import os
sys.path.append(os.path.join(sys.path[0], '../EleutherAI_Eval_harness'))
sys.path.append(os.path.join(sys.path[0], '../'))

from lm_eval.models.mamba_models_EE import Mamba_7b
from lm_eval import evaluate, simple_evaluate
from lm_eval.tasks import get_task_dict
from lm_eval.utils import make_table

# os.chdir(os.path.join(sys.path[0], './EE_Clean'))
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from models.mamba.models.wrapper_mamba_EE import Mamba
from models.mamba.mistral_inference.args import MambaArgs
from models.mistral.tokenizer import Tokenizer

import time
import json
import torch


path_weights = "./weights/mamba/mamba-codestral-7B-v0.1"
max_length = 2048*2
max_gen_tokens = 32
device = "cuda"
batch_size = 1
recompute_states = False


print("Loading model...")

path = f"./weights/mamba"
path_weigths_EE = path + f"/EE_1_layers_middle_2_wsum_pos_31_39_47_55"
plot_intermediate_states = True
th_for_EE = 0.5
ee_pos = [int(p) for p in path_weigths_EE.split("_pos_")[-1].split("_")]

# Load model arguments from JSON
with open("./weights/mamba/mamba-codestral-7B-v0.1/params.json", "r") as f:
    model_args = MambaArgs.from_dict(json.load(f))
    print(model_args)

# Assign ee_pos and other model arguments
model_args.ee_pos = ee_pos
model_args.block_size = max_length

# Initialize the model
model = Mamba(model_args)

# Start timing for model loading
start_time = time.time()
model.from_folder("./weights/mamba/mamba-codestral-7B-v0.1")
print(f"Time to load model: {time.time() - start_time}")

# Update layer settings and thresholds if necessary
n_layer = model_args.n_layers
model.th = model.th * th_for_EE if ee_pos is not None else None

# Lazy initialization of GPU if using CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Sending model to GPU...")
start_time = time.time()

# Ensure the model is in eval mode and move to the device with bf16 precision
model.eval()
if device.type == 'cuda':
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        model.to(device=device)
    print(f"Time to load model to GPU: {time.time() - start_time}")

# Load EE layer weights gradually to avoid memory spikes
if ee_pos is not None:
    for i in range(len(ee_pos)):
        layer_path = f"{path_weigths_EE}/layer_{i}_EE"
        print(f"Loading EE weights for layer {i} from {layer_path}...")
        state_dict = torch.load(layer_path, map_location=device)
        model.model.backbone.ee[i].load_state_dict(state_dict)
        print(f"Layer {i} EE weights loaded.")

print("Model successfully loaded and sent to GPU.")

print("Loading tokenizer...")
tokenizer = Tokenizer("./weights/mamba/mamba-codestral-7B-v0.1/tokenizer.model.v3")
# tokenizer = MistralTokenizer.v3().instruct_tokenizer


datasets = ["triviaqa", "coqa", "truthfulqa_gen"]
# datasets = ["truthfulqa_gen"]
n_shots = [2, 0, 1]
ranges_th = torch.tensor([[0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.45, 0.5, 0.7, 1],
                         [0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.45, 0.5, 0.7, 1],
                         [0.26, 0.27, 0.28, 0.29, 0.3, 0.35, 0.4, 0.5, 0.6, 1]]
                        )
ranges_blocks = torch.tensor([[1, 2, 3, 4, 5, -1],
                              [1, 4, 6, 10, 12, 24],
                              [1, 4, 6, 10, 12, 24]]
                             )
for dataset,fewshots_set, range_th, range_blocks in zip(datasets, n_shots, ranges_th, ranges_blocks):

    
    results_list = []
    exits_done = []
    positions_exited = []
    lens_generated = []
    recompute_states = True
    for th in range_th:

        if th == -1:
            continue

        print("Evaluate for th = ", th, " for ", dataset, " with recompute states")

        model.th = torch.ones(n_layer - 1) * th

        lm_obj = Mamba_7b(  model=model,
                    tokenizer = tokenizer,
                    batch_size=batch_size,
                    max_length = max_length,
                    max_gen_tokens = max_gen_tokens if th not in [0, 0.1, 0.2] else round(max_gen_tokens/2),
                    temperature = 1.0,
                    top_k = None,
                    recompute_states = recompute_states,
                    use_EE = True if ee_pos is not None else False,
                    device = device)

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

        print(make_table(results))

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

    # print(make_table(results_list[0]))

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

           
    results_list = []
    exits_done = []
    positions_exited = []
    lens_generated = []
    recompute_states = False
    for th in range_th:

        if th == -1:
            continue

        print("Evaluate for th = ", th, " for ", dataset, " without recompute states")

        model.th = torch.ones(n_layer - 1) * th

        lm_obj = Mamba_7b(  model=model,
                    tokenizer = tokenizer,
                    batch_size=batch_size,
                    max_length = max_length,
                    max_gen_tokens = max_gen_tokens if th not in [0, 0.1, 0.2] else round(max_gen_tokens/2),
                    temperature = 1.0,
                    top_k = None,
                    recompute_states = recompute_states,
                    use_EE = True if ee_pos is not None else False,
                    device = device)

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

        print(make_table(results))

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

    # print(make_table(results_list[0]))

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


    results_list = []
    for n_block in range_blocks:

        if n_block == -1:
            continue
        
        drops = range_blocks

        print("Evaluate baseline for ", n_block, " blocks dropped for ", dataset)
            
        lm_obj = Mamba_7b(model=model,
                            tokenizer = tokenizer,
                            batch_size=batch_size,
                            max_length = max_length,
                            max_gen_tokens = max_gen_tokens,
                            temperature = 1.0,
                            top_k = None,


                            use_EE = False,
                            n_blocks = n_layer - n_block,
                            device = device)

        print("Evaluating...")

        results = simple_evaluate(
            model = lm_obj,
            tasks = [dataset],
            num_fewshot = fewshots_set,
        )

        results_list.append(results)


        print(make_table(results))


    # save results list, the exits done and the positions, if it does not exist, create it
    if not os.path.exists(path_weigths_EE + f"/results/" + dataset + "/baseline"):
        os.makedirs(path_weigths_EE + f"/results/" + dataset + "/baseline")

    with open(path_weigths_EE + f"/results/"+dataset+ "/baseline" +"/results_list.json", "w") as f:
        json.dump(results_list, f)

    with open(path_weigths_EE + f"/results/"+dataset+ "/baseline"+"/layers_dropped.json", "w") as f:
        json.dump(drops.tolist(), f)

