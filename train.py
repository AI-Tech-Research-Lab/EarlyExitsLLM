from datasets import load_dataset
import torch

from utils import get_model_and_tok_train

import os

from schedulefree import AdamWScheduleFree

# use name="sample-10BT" to use the 10BT sample
fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

device = "cuda"
model_choice = "mamba"


# size = "350" # 124M, 350M, 774M, 1558M
# path = f"./weights/gpt2/gpt2_{size}M_100B_FinewebEdu_hf"
# path = f"./weights/gpt2/gpt2"
# path = f"./weights/mistral"
path = f"./weights/mamba"


path_weigths_EE = path + f"./EE_1_layers_middle_2"
train_type = "wsum" # wsum, individual
path_weigths_EE = path_weigths_EE + f"_{train_type}"

ee_pos = [31, 39, 47, 55]
# ee_pos = [15, 19, 23, 27]

model, encode, decode = get_model_and_tok_train(model_choice, path, ee_pos, device)

model.freeze_backbone()

batch_size = 16
lr = 2.5e-4

fw = fw.shuffle()

# check number of trainble params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print("Trainable parameters", trainable)
print(" EEs are a", round(100*trainable/total, 5), "% of the", model_choice, "model")

val_freq = 10

optimizers = []
for i in range(len(ee_pos)):
    if model_choice == "gpt2":
        optimizers.append(AdamWScheduleFree(model.transformer.h[i].ee.parameters(), lr=lr))
    elif model_choice == "mistral":
        optimizers.append(AdamWScheduleFree(model.ee[i].parameters(), lr=lr))
    elif model_choice == "mamba":
        optimizers.append(AdamWScheduleFree(model.model.backbone.ee[i].parameters(), lr=lr))

iters = 1000
model.k = 15
metrics_val, metrics = torch.zeros((int(iters/val_freq), 5)), torch.zeros((int(iters-iters/val_freq),5))


# create training loop
for i, ex in enumerate(fw):

    if iters == i:
        break

    # if i == 0:
    e = encode(ex["text"])
        # e = encode("bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang")

    # crop long inputs
    if model_choice == "gpt2":
        if e.size(1) > 1024:
            e = e[:, :1024]
    if model_choice == "mistral" or model_choice == "mamba":
        if e.size(0) > 1024*4:
            e = e[:1024*4]
        if model_choice == "mamba":
            model.args.block_size = 1024*4

    if i % val_freq == 0:
        with torch.no_grad():
            _, loss_val, metrics_val[int(i/val_freq)] = model(e, train_EE = True)
        print(f"Validation loss: {loss_val[0].item():.4f}")

    else:

        _, loss, metrics[i - int(i/val_freq) - 1] = model(e, train_EE = True)

        if train_type == "individual":
            
            for i in range(len(optimizers)):
                loss[i].backward(retain_graph=True)

        elif train_type == "wsum":
            
            w = torch.arange(1, len(optimizers)+1)/sum(torch.arange(1, len(optimizers)+1))

            loss_wsum = (loss * w).sum()

            loss_wsum.backward()            

        for optimizer in optimizers:

            optimizer.step()

            optimizer.zero_grad()
            
        # losses = [l.item() for l in loss]
        # print(f"Training loss: {losses}")

        # print(f"Training loss: {loss.item():.4f}")

# test run
test_iters = 10
metrics_test = torch.zeros((test_iters, 5))
max_length = 128
print("Running test...")
for i, ex in enumerate(fw):

    if i == test_iters:
        break

    e = encode(ex["text"])

    # crop long inputs
    if model_choice == "gpt2":
        if e.size(1) > 1024:
            e = e[:, :1024]
    if model_choice == "mistral" or model_choice == "mamba":
        if e.size(0) > min([1024*4, max_length]):
            e = e[:min([1024*4,max_length])]
        if model_choice == "mamba":
            model.args.block_size = 1024*4

    with torch.no_grad():
        _, _, metrics_test[i,:] = model(e, train_EE = True)

print("Test metrics:")
print("Test Accuracy:", metrics_test[:,0].mean().item())
print("Test Recall:", metrics_test[:,1].mean().item())
print("Test Precision:", metrics_test[:,2].mean().item())
print("Test F1:", metrics_test[:,3].mean().item())


import matplotlib.pyplot as plt

# generate x axis for val and train
x_train = torch.arange(0, iters, 1)
x_val = x_train[::val_freq]
x_train = [p.item() for p in x_train if p not in x_val]

print("Mean ratio is:", metrics[:,-1].mean().item())


print("Saving weights")

# save EE weights
if model_choice == "gpt2":
    i = -1
    for n,m in model.transformer.h[0].ee.named_modules():
        i += 1
    name = f"EE_{i}_layers_middle_{model.transformer.h[0].ee.c_fc.weight.size(0)}"

    # create folder with name
    if not os.path.exists(path +f"/{name}"):
        os.makedirs(path +  f"/{name}")

    for i in range(n_layer - 1):
        torch.save(model.transformer.h[i].ee.state_dict(), path + f"/{name}/layer_{i}_EE")


# save EE weights
# if model_choice == "mistral":
#     i = -1
#     for n,m in model.layers[0].ee.named_modules():
#         i += 1
#     name = f"EE_{i}_layers_middle_{model.layers[0].ee.c_fc.weight.size(0)}"

#     # create folder with name
#     if not os.path.exists(f"./weights/mistral/{name}"):
#         os.makedirs(f"./weights/mistral/{name}")

#     for i in range(n_layer - 1):
#         torch.save(model.layers[i].ee.state_dict(), f"./weights/mistral/{name}/layer_{i}_EE")

# save EE weights
if model_choice == "mistral":
    i = -1
    for n,m in model.ee[0].named_modules():
        i += 1
    name = f"EE_{i}_layers_middle_{model.ee[0].c_fc.weight.size(0)}_{train_type}_pos_{'_'.join(map(str, ee_pos))}"
    # create folder with name
    if not os.path.exists(f"./weights/mistral/{name}"):
        os.makedirs(f"./weights/mistral/{name}")

    ee_index = 0
    for i in range(len(ee_pos)):
        torch.save(model.ee[0].state_dict(), f"./weights/mistral/{name}/layer_{ee_index}_EE")
        ee_index += 1

# save EE weights
if model_choice == "mamba":
    i = -1
    for n,m in model.model.backbone.ee[0].named_modules():
        i += 1
    name = f"EE_{i}_layers_middle_{model.model.backbone.ee[0].c_fc.weight.size(0)}_{train_type}_pos_{'_'.join(map(str, ee_pos))}"


    # create folder with name
    if not os.path.exists(f"./weights/mamba/{name}"):
        os.makedirs(f"./weights/mamba/{name}")

    ee_index = 0
    for i in range(len(ee_pos)):
        torch.save(model.model.backbone.ee[0].state_dict(), f"./weights/mamba/{name}/layer_{ee_index}_EE")
        ee_index += 1

import matplotlib.pyplot as plt
import torch

# Creating the figure with a larger size for better readability
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Training and Validation Metrics', fontsize=16)

# List of metric names for easier looping
metric_names = ['EE Accuracy', 'EE Recall', 'EE Precision', 'EE F1']
windows = []
window = int(iters*0.05)
for i in range(4):
    windowed_mean = torch.tensor([metrics[j:j+window, i].mean().item() for j in range(iters-window)])
    windows.append(windowed_mean)

# Plotting each metric
for i, ax in enumerate(axs.flat):
    ax.plot(x_train, metrics[:, i].numpy(), label="Training", linewidth=1.5, color="blue")
    ax.plot(x_val, metrics_val[:, i].numpy(), label="Validation", linewidth=1.5, color="orange")

    # Plot the windowed mean in red
    ax.plot(torch.arange(window/2, iters-window + window/2, 1), windows[i], 
            label="Windowed Mean", color="red", linewidth=2.5)

    # Set title, grid and legend
    ax.set_title(metric_names[i], fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_ylim([0.5, 1])  # Assuming the metric is between 0.5 and 1 for better visual range

# Apply tight layout for better spacing and positioning
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
# plt.show()
# save plot with weights of EE
if model_choice == "gpt2":
    plt.savefig(f"./weights/gpt2/{name}/metrics.png")
elif model_choice == "mistral":
    plt.savefig(f"./weights/mistral/{name}/metrics.png")
elif model_choice == "mamba":
    plt.savefig(f"./weights/mamba/{name}/metrics.png")
