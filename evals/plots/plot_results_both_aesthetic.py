import json
import os
import sys
sys.path.append(os.path.join(sys.path[0], '../../../'))
# Define the path to the folder where the files are located
# path = f"./weights/mistral"
# path_weights_EE = path + f"/EE_1_layers_middle_2_wsum_pos_15_19_23_27"
# dataset = "truthfulqa_gen"
# submetric = "max" # acc, diff, max
# baseline = True

path = "./weights/mamba"
path_weights_EE = path + f"/EE_1_layers_middle_2_wsum_pos_31_39_47_55"
dataset = "truthfulqa_gen"
recomputation = True
submetric = "max" # acc, diff, max
baseline = True

recomp = "/recompute_states"
norecomp = "/no_recomp" 
penalize_mistral = 4/24
penalize_mamba = 9/26

# Load the results_list from the JSON file
with open(f"{path_weights_EE}/results/"+dataset+recomp+"/results_list.json", "r") as f:
    results_list = json.load(f)

# Load the exits_done from the JSON file
with open(f"{path_weights_EE}/results/"+dataset+recomp+"/exits_done.json", "r") as f:
    exits_done = json.load(f)

# Load the positions_exited from the JSON file
with open(f"{path_weights_EE}/results/"+dataset+recomp+"/positions_exited.json", "r") as f:
    positions_exited = json.load(f)

with open(f"{path_weights_EE}/results/"+dataset+recomp+"/lens_generated.json", "r") as f:
    lens_generated = json.load(f)

with open(f"{path_weights_EE}/results/"+dataset+norecomp+"/results_list.json", "r") as f:
    results_list_norecomp = json.load(f)

with open(f"{path_weights_EE}/results/"+dataset+norecomp+"/exits_done.json", "r") as f:
    exits_done_norecomp = json.load(f)

with open(f"{path_weights_EE}/results/"+dataset+norecomp+"/positions_exited.json", "r") as f:
    positions_exited_norecomp = json.load(f)

with open(f"{path_weights_EE}/results/"+dataset+norecomp+"/lens_generated.json", "r") as f:
    lens_generated_norecomp = json.load(f)

with open(f"{path_weights_EE}/results/"+dataset+recomp+"/th_swept.json", "r") as f:
    range_th = json.load(f)


if baseline:
    # open the baseline from {path_weights_EE}/results/"+dataset+"baseline"
    with open(f"{path_weights_EE}/results/"+dataset+"/baseline/results_list.json", "r") as f:
        results_list_baseline = json.load(f)

    with open(f"{path_weights_EE}/results/"+dataset+"/baseline/layers_dropped.json", "r") as f:
        layers_dropped_baseline = json.load(f)

# # Now the variables results_list, exits_done, and positions_exited are loaded and ready to use
# print("Results List:", results_list)
# print("Exits Done:", exits_done)
# print("Positions Exited:", positions_exited)
# print("Lens Generated:", lens_generated)

# plot a graph of the results 
import matplotlib.pyplot as plt
import torch

n_layers = 32 if "mistral" in path_weights_EE else 64

range_th = torch.tensor(range_th)

# compute stimated speedup
speedups_recomp = []
speedups_norecomp = []
speedup_r = 0
skip_recomp = []
skip_norecomp = []
threshold_deg_res = 100 if dataset == "truthfulqa_gen" else 35
threshold_deg_res_mamba = 411 if dataset == "truthfulqa_gen" else 100
if "mistral" in path_weights_EE:
    for i, th in enumerate(range_th):
        # recomp
        if exits_done[i] == [] or exits_done[i] == 0:
            speedups_recomp.append(1)
        else:
            # for ex, pos, len in zip(exits_done, positions_exited, lens_generated):
            # speedup_r += n_layers/torch.tensor(ex, dtype = torch.float).mean()
            lens = torch.tensor(lens_generated[i], dtype = torch.float)
            deg_res = sum(lens == -1)
            lens[lens == -1] = (lens+1).mean()


            print(deg_res)
            if deg_res > threshold_deg_res:
                skip_recomp.append(i)
            else:
                total_blocks = sum(torch.tensor(lens_generated[i], dtype = torch.float) - 1) * n_layers
                blocks_ignored = sum(n_layers - torch.tensor(exits_done[i], dtype = torch.float) + 1)
                pen = sum(n_layers - torch.tensor(exits_done[i], dtype = torch.float) + 1)*penalize_mistral
                # pen = deg_res*min(exits_done[i])*n_layers
                # pen = 0
                
                speedup_r = total_blocks / (total_blocks - blocks_ignored + pen)
                speedups_recomp.append(speedup_r)

        if exits_done_norecomp[i] == [] or exits_done_norecomp[i] == 0:
            speedups_norecomp.append(1)
        else:
            lens = torch.tensor(lens_generated_norecomp[i], dtype = torch.float)
            deg_res = sum(lens == -1)
            lens[lens == -1] = (lens+1).mean()


            if deg_res > threshold_deg_res:
                skip_norecomp.append(i)
            else:
                total_blocks = sum(torch.tensor(lens_generated_norecomp[i], dtype = torch.float) - 1) * n_layers
                blocks_ignored = sum(n_layers - torch.tensor(exits_done_norecomp[i], dtype = torch.float))

                speedup_r = total_blocks / (total_blocks - blocks_ignored)
                speedups_norecomp.append(speedup_r)


elif "mamba" in path_weights_EE:
    for i, th in enumerate(range_th):

        if exits_done[i] == [] or exits_done[i] == 0:
            speedups_recomp.append(1)
        else:
            lens = torch.tensor(lens_generated[i], dtype = torch.float)
            deg_res = sum(lens == -1)
            lens[lens == -1] = 32 if dataset == "truthfulqa_gen" else lens.mean()

            print(deg_res)

            if deg_res > threshold_deg_res_mamba:
                skip_recomp.append(i)
            else:
                total_blocks = sum(torch.tensor(lens, dtype = torch.float) - 1) * n_layers
                blocks_ignored = sum(n_layers - torch.tensor(exits_done[i], dtype = torch.float))
                pen = sum(n_layers - torch.tensor(exits_done[i], dtype = torch.float) + 1)*penalize_mamba
                # pen = deg_res*min(exits_done[i])*n_layers
                # pen = 0
                
                speedup_r = total_blocks / (total_blocks - blocks_ignored + pen)
                speedups_recomp.append(speedup_r)


        if exits_done_norecomp[i] == [] or exits_done_norecomp[i] == 0:
            speedups_norecomp.append(1)
        else:
            lens = torch.tensor(lens_generated_norecomp[i], dtype = torch.float)
            deg_res = sum(lens == -1)
            lens[lens == -1] = 32 if dataset == "truthfulqa_gen" else lens.mean()
            if deg_res > threshold_deg_res_mamba:
                skip_norecomp.append(i)
            else:
                total_blocks = sum(torch.tensor(lens, dtype = torch.float) - 1) * n_layers
                blocks_ignored = sum(n_layers - torch.tensor(exits_done_norecomp[i], dtype = torch.float))

                speedup_r = total_blocks / (total_blocks - blocks_ignored)
                speedups_norecomp.append(speedup_r)
        
        
            # if recomp:
            #     pen = (penalize*(n_layers-torch.tensor(exits_done[i], dtype = torch.float).mean()))
            # else:
            #     pen = 0
            # speedups.append(n_layers/(pen+torch.tensor(exits_done[i], dtype = torch.float).mean()))

# Define the y-axis values
if dataset == "triviaqa":
    metrics = ["exact_match,remove_whitespace"]
elif dataset == "coqa":
    metrics = ["f1,none", "em,none"]
elif dataset == "truthfulqa_gen":
    metrics = ["bleu_"+submetric+",none","rouge1_"+submetric+",none","rouge2_"+submetric+",none","rougeL_"+submetric+",none"]

metric_values = []
metric_values_norecomp = []
bl_values = []
for i,metric in enumerate(metrics):
    metric_values.append([r["results"][dataset][metric] for i,r in enumerate(results_list) if i not in skip_recomp])
    metric_values_norecomp.append([r["results"][dataset][metric] for i,r in enumerate(results_list_norecomp) if i not in skip_norecomp])
    if baseline:
        bl_values.append([r["results"][dataset][metric] for r in results_list_baseline])

# filter out bl_values where layers_dropped_baseline is -1
bl_clean = []
if baseline:
    for j,metric in enumerate(metrics):
        bl_clean.append([bl for i, bl in enumerate(bl_values[j]) if layers_dropped_baseline[i] != -1])
bl_values = bl_clean
layers_dropped_baseline = [l for l in layers_dropped_baseline if l != -1]

range_th_recomp = [th for i, th in enumerate(range_th) if i not in skip_recomp]
range_th_norecomp = [th for i, th in enumerate(range_th) if i not in skip_norecomp]


import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# Update matplotlib rcParams for LaTeX rendering
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
})

# sns.set_theme(style="whitegrid")  # Apply Seaborn style to the plots

for j, (metric_r, metric_nr) in enumerate(zip(metric_values, metric_values_norecomp)):
    plt.figure(figsize=(10, 5)) 

    # Set legend based on metric type
    if "exact_match" in metrics[j]:
        legend = "Exact Match"
    elif "f1" in metrics[j]:
        legend = "F1"
    elif "em" in metrics[j]:
        legend = "Exact Match"
    elif "bleu" in metrics[j]:
        legend = "BLEU " + submetric
    elif "rouge1" in metrics[j]:
        legend = "ROUGE-1 " + submetric
    elif "rouge2" in metrics[j]:
        legend = "ROUGE-2 " + submetric
    elif "rougeL" in metrics[j]:
        legend = "ROUGE-L " + submetric

    # Normalize the colors array for proper color mapping
    norm = plt.Normalize(vmin=min(range_th_norecomp), vmax=max(range_th_norecomp))
    cmap = sns.color_palette("winter", as_cmap=True)  # Use Seaborn color palette for a smoother gradient

    # Plot lines
    plt.plot(speedups_recomp, metric_r, color='#0BA4FF', zorder=1, label="Early exits Recomp")
    plt.plot(speedups_norecomp, metric_nr, color='#AEDD00', zorder=1, label="Early exits No Recomp")

    # Plot points with colors
    scatter = plt.scatter(speedups_recomp, metric_r, c=range_th_recomp, cmap=cmap, norm=norm, marker='D')
    plt.scatter(speedups_norecomp, metric_nr, c=range_th_norecomp, cmap=cmap, norm=norm, marker='D')

    # Add colorbar
    plt.colorbar(scatter, label='Threshold Value')

    # Baseline as scatter
    if baseline:
        plt.scatter(n_layers / (n_layers - torch.tensor(layers_dropped_baseline)), bl_values[j], color='#FF8C8C', zorder=2, label="Layer Pruning")

    plt.xlabel('Computational Reduction Factor', weight='bold')
    plt.ylabel(legend, weight='bold')

    # Dataset name assignment
    dataset_name = {"truthfulqa_gen": "TruthfulQA", "triviaqa": "TriviaQA", "coqa": "CoQA"}.get(dataset, dataset)

    plt.legend(fancybox=True, framealpha=1, shadow=True)

    plt.title(f"Speedup vs {legend} for {dataset_name} Dataset")
    model_name = "mistral" if "mistral" in path_weights_EE else "mamba"
    plt.grid(visible=True, linestyle=':', linewidth=0.7)


    # Save individual plot
    plt.savefig(f"{path_weights_EE}/results/{dataset}/final_{model_name}_speedup_vs_{metrics[j].split(',')[0]}_{dataset}.png", dpi=400)

# Generate combined figures based on dataset
if dataset == "coqa":
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for j, (metric_r, metric_nr) in enumerate(zip(metric_values, metric_values_norecomp)):
        ax = axs[j]
        metric_legend = "F1" if j == 0 else "Exact Match"
        
        ax.plot(speedups_recomp, metric_r, color='#0BA4FF', zorder=1, label="Early exits Recomp")
        ax.plot(speedups_norecomp, metric_nr, color='#AEDD00', zorder=1, label="Early exits No Recomp")
        ax.scatter(speedups_recomp, metric_r, c=range_th_recomp, cmap=cmap, norm=norm, marker='D')
        ax.scatter(speedups_norecomp, metric_nr, c=range_th_norecomp, cmap=cmap, norm=norm, marker='D')
        
        if baseline:
            ax.scatter(n_layers / (n_layers - torch.tensor(layers_dropped_baseline)), bl_values[j], color='#FF8C8C', zorder=2, label="Layer Pruning")

        ax.set_xlabel('Computational Reduction Factor', weight='bold')
        ax.set_ylabel(metric_legend, weight='bold')
        ax.set_title(f"Speedup vs {metric_legend}")
        ax.grid(visible=True, linestyle=':', linewidth=0.7)

        ax.legend(fancybox=True, framealpha=1, shadow=True)


    plt.tight_layout(rect=[0, 0, 0.85, 1])

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(scatter, cax=cax, label='Threshold Value')

    plt.savefig(f"{path_weights_EE}/results/{dataset}/final_{model_name}_speedup_vs_combined_{dataset}.png", dpi=400)

if dataset == "truthfulqa_gen":
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for j, (metric_r, metric_nr) in enumerate(zip(metric_values, metric_values_norecomp)):
        ax = axs[j // 2, j % 2]
        metric_legend = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"][j]

        ax.plot(speedups_recomp, metric_r, color='#0BA4FF', zorder=1, label="Early exits Recomp")
        ax.plot(speedups_norecomp, metric_nr, color='#AEDD00', zorder=1, label="Early exits No Recomp")
        ax.scatter(speedups_recomp, metric_r, c=range_th_recomp, cmap=cmap, norm=norm, marker='D')
        ax.scatter(speedups_norecomp, metric_nr, c=range_th_norecomp, cmap=cmap, norm=norm, marker='D')

        if baseline:
            ax.scatter(n_layers / (n_layers - torch.tensor(layers_dropped_baseline)), bl_values[j], color='#FF8C8C', zorder=2, label="Layer Pruning")

        ax.set_xlabel('Computational Reduction Factor', weight='bold')
        ax.set_ylabel(metric_legend, weight='bold')
        ax.set_title(f"Speedup vs {metric_legend}")
        ax.grid(visible=True, linestyle=':', linewidth=0.7)
        ax.legend(fancybox=True, framealpha=1, shadow=True)
        

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Threshold Value')

    plt.savefig(f"{path_weights_EE}/results/{dataset}/final_{submetric}_{model_name}_speedup_vs_combined_{dataset}.png", dpi=500)
