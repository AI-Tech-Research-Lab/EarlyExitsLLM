
from models.mistral.model import ModelArgs
from models.mistral.model_EE import Transformer
from models.mistral.tokenizer import Tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


import json
import torch

path_weights = "./weights/mistral/7B-v0.3"
max_length = 2048*2
max_gen_tokens = 64
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
path_weigths_EE = path + f"/EE_1_layers_middle_2_wsum_pos_15_19_23_27"
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


k = 50
least = []
most = []
for i,ee in enumerate(ee_pos):
    weigths = model.ee[i].c_fc.weight
    print(weigths)

    # run weights through model head
    n_w = model.norm(weigths)
    print(n_w)

    # run weights through model head
    n_w = model.output(n_w)

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