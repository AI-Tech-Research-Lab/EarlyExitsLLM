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

path_weights = "./weights/mamba/mamba-codestral-7B-v0.1"
max_length = 2048*2
max_gen_tokens = 32
device = "cuda"
batch_size = 4
recompute_states = False


import time
import torch

print("Loading model...")

path = f"./weights/mamba"
path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_32_40_48_56"
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


dataset = "mmlu"
n_shots = 5

# Create directories if they do not exist
output_dir = f"{path_weigths_EE}/results/{dataset}/baseline"
os.makedirs(output_dir, exist_ok=True)

# Define the directory and filenames
output_dir = f"{path_weigths_EE}/results/{dataset}/baseline"
results_file = f"{output_dir}/{n_shots}_results_list.json"
drops_file = f"{output_dir}/{n_shots}_layers_dropped.json"
os.makedirs(output_dir, exist_ok=True)

# Load previous progress if files exist
if os.path.exists(results_file) and os.path.exists(drops_file):
    with open(results_file, "r") as f:
        results_buffer = json.load(f)
    with open(drops_file, "r") as f:
        drops_buffer = json.load(f)
else:
    results_buffer = []
    drops_buffer = []

# Determine starting point based on the length of previously saved data
start_iter = len(results_buffer)
last_iter = 64
drops = torch.arange(start_iter, last_iter, 1)
for layers_dropped in drops:

    print(f"Layers dropped: {layers_dropped}")


    lm_obj = Mamba_7b(model=model,
                        tokenizer = tokenizer,
                        batch_size=batch_size,
                        max_length = max_length,
                        max_gen_tokens = max_gen_tokens,
                        temperature = 1.0,
                        top_k = None,
                        use_EE = False,
                        n_blocks = n_layer - layers_dropped,
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
        tasks = [dataset],
        num_fewshot = n_shots,
    )

    results_buffer.append(results)
    drops_buffer.append(layers_dropped.item())

    print(make_table(results))

    # Append new result to JSON file each iteration
    with open(f"{output_dir}/{n_shots}_results_list.json", "w") as f:
        f.write(json.dumps(results_buffer))  # Append as new line for each result
    
    with open(f"{output_dir}/{n_shots}_layers_dropped.json", "w") as f:
        f.write(json.dumps(drops_buffer))  # Append as new line for each result

# # save results list, the exits done and the positions, if it does not exist, create it
# if not os.path.exists(path_weigths_EE + f"/results/" + dataset + "/baseline"):
#     os.makedirs(path_weigths_EE + f"/results/" + dataset + "/baseline")

# with open(path_weigths_EE + f"/results/"+dataset+ "/baseline" +"/" + str(n_shots) + "_results_list.json", "w") as f:
#     json.dump(results_list.tolist(), f)

# with open(path_weigths_EE + f"/results/"+dataset+ "/baseline"+"/" + str(n_shots) + "_layers_dropped.json", "w") as f:
#     json.dump(drops.tolist(), f)
