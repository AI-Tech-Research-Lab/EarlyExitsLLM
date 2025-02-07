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
path_weigths_EE = path + f"/EE_1_layers_middle_2_wsum_pos_31_39_47_55"
plot_intermediate_states = True
th_for_EE = 0.5
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


k = 50
least = []
most = []
for i,ee in enumerate(ee_pos):
    weigths = model.model.backbone.ee[i].c_fc.weight
    print(weigths)

    # run weights through model head
    n_w = model.model.backbone.norm_f(weigths[:,:4096])
    print(n_w)

    # run weights through model head
    n_w = model.model.lm_head(n_w.to(torch.bfloat16))

    logits_least = n_w[0]
    logits_most = n_w[1]

    # get 10 argmax
    values1,toks1 = torch.topk(logits_least, k)
    values2,toks2 = torch.topk(logits_most, k)

    # decode
    for i,tok in enumerate(toks1):
        word = tokenizer._model.decode(tok.tolist())
        print("Least likely token", str(tok.item()),"number", str(i), word, "with value", values1[i].item())
        least.append((i, word, values1[i].item()))

    print("\n")
    print("%"*50)
    print("\n")

    for i,tok in enumerate(toks2):
        word = tokenizer._model.decode(tok.tolist())
        print("Most likely token", str(tok.item()),"number", str(i), word, "with value", values2[i].item())
        most.append((i, word, values2[i].item()))

    print("%"*50)
    print("\n"*2)

    # % e is 1085

# plot histogram of logits_most and logits_least
import matplotlib.pyplot as plt
import numpy as np


logits_least = logits_least.to(torch.float32)
logits_most = logits_most.to(torch.float32)
plt.hist(logits_least.cpu().detach().numpy(), bins=100, alpha=0.5, label='Least likely')
plt.hist(logits_most.cpu().detach().numpy(), bins=100, alpha=0.5, label='Most likely')
plt.legend(loc='upper right')
plt.savefig("histogram.png")


# save the most and least likely tokens
import csv
with open(path_weigths_EE+'/most.csv', mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(["Rank", "Token", "Value"])
    for row in most:
        writer.writerow(row)

with open(path_weigths_EE+'/least.csv', mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(["Rank", "Token", "Value"])
    for row in least:
        writer.writerow(row)