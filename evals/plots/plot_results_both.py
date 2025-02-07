import json
import os
import sys
sys.path.append(os.path.join(sys.path[0], '../../../'))
# Define the path to the folder where the files are located
# path = f"./weights/mistral"
# path_weights_EE = path + f"/EE_1_layers_middle_2_wsum_pos_15_19_23_27"
# dataset = "truthfulqa_gen"
# submetric = "diff" # acc, diff, max
# baseline = True

path = "./weights/mamba"
path_weights_EE = path + f"/EE_1_layers_middle_2_wsum_pos_31_39_47_55"
dataset = "triviaqa"
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
for j, (metric_r, metric_nr) in enumerate(zip(metric_values, metric_values_norecomp)):
    # plot speedups vs y
    plt.figure(figsize=(10, 5)) 

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
    cmap = plt.get_cmap('winter')

    # # Define your colors: #102C53 for value 1, white for value 0
    # colors = ['#FFFFFF', '#102C53']
    # # Create a colormap
    # cmap = LinearSegmentedColormap.from_list('custom_cmap', colors) 

    # Plot the line (without color)
    plt.plot(speedups_recomp, metric_r, color='#0BA4FF', zorder=1, label = "Early exits Recomp")
    plt.plot(speedups_norecomp, metric_nr, color='#AEDD00', zorder=1, label = "Early exits No Recomp")


    # Plot the points with colors
    scatter = plt.scatter(speedups_recomp, metric_r, c=range_th_recomp, cmap=cmap, norm=norm, marker='D')
    scatter = plt.scatter(speedups_norecomp, metric_nr, c=range_th_norecomp, cmap=cmap, norm=norm, marker='D')

    # Add colorbar to the plot
    plt.colorbar(scatter, label='Threshold Value')

    # add the baseline as scatter
    if baseline:
        plt.scatter(n_layers/(n_layers-(torch.tensor(layers_dropped_baseline))), bl_values[j], color='#FF8C8C', zorder=2, label = "Layer Pruning")

    plt.xlabel('Computational reduction factor')
    plt.ylabel(legend)
    # plt legend, baseline should be appear only once
    
    if dataset == "truthfulqa_gen":
        dataset_name = "TruthfulQA"
    elif dataset == "triviaqa":
        dataset_name = "TriviaQA"
    elif dataset == "coqa":
        dataset_name = "CoQA"

    plt.legend()
    plt.title(f"Speedup vs {legend} for {dataset_name} dataset")  
    model_name = "mistral" if "mistral" in path_weights_EE else "mamba"
    plt.grid()


    plt.savefig(f"{path_weights_EE}/results/"+dataset+"/final_"+model_name+"_speedup_vs_"+metrics[j].split(",")[0]+"_"+dataset+".png", dpi = 400)
    
    # check if the folder exists, if not create it
    # import os
    # if not os.path.exists(f"./TEST/{path_weights_EE[1:]}/results/"+dataset+recomp):
    #     os.makedirs(f"./TEST/{path_weights_EE[1:]}/results/"+dataset+recomp)
    # plt.savefig(f"./TEST/{path_weights_EE[1:]}/results/"+dataset+recomp+"/"+model_name+"_speedup_vs_"+metrics[j].split(",")[0]+".png")
  
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Add the functionality to combine both figures in a subfigure if dataset is "coqa"
# Add the functionality to combine both figures in a subfigure if dataset is "coqa"
if dataset == "coqa":
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # Create two subplots in a single figure
    
    for j, (metric_r, metric_nr) in enumerate(zip(metric_values, metric_values_norecomp)):
        if j == 0:
            ax = axs[0]
            metric_legend = "F1"
        else:
            ax = axs[1]
            metric_legend = "Exact Match"
        
        # Normalize the colors array for proper color mapping
        norm = plt.Normalize(vmin=min(range_th_norecomp), vmax=max(range_th_norecomp))
        cmap = plt.get_cmap('winter')
        
        # Plot the line for recomputation and no recomputation
        ax.plot(speedups_recomp, metric_r, color='#0BA4FF', zorder=1, label="Early exits Recomp")
        ax.plot(speedups_norecomp, metric_nr, color='#AEDD00', zorder=1, label="Early exits No Recomp")

        # Plot the points with colors
        scatter = ax.scatter(speedups_recomp, metric_r, c=range_th_recomp, cmap=cmap, norm=norm, marker='D')
        ax.scatter(speedups_norecomp, metric_nr, c=range_th_norecomp, cmap=cmap, norm=norm, marker='D')

        # Add the baseline as scatter points
        if baseline:
            ax.scatter(n_layers / (n_layers - torch.tensor(layers_dropped_baseline)), bl_values[j], 
                       color='#FF8C8C', zorder=2, label="Layer Pruning")

        ax.set_xlabel('Computational reduction factor')
        ax.set_ylabel(metric_legend)
        ax.grid()
        ax.set_title(f"Speedup vs {metric_legend}")

        # Add individual legends back to each subplot
        ax.legend()

    # Adjust layout to ensure space for colorbar and prevent overlapping
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust to leave space for colorbar on the right

    # Create a divider for colorbar in the second axis (Exact Match plot)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and padding for the colorbar
    
    # Add the colorbar to the right of the figure (shared across both subplots)
    fig.colorbar(scatter, cax=cax, label='Threshold Value')

    # Save the combined subfigure
    plt.savefig(f"{path_weights_EE}/results/" + dataset + "/final_" + model_name + "_speedup_vs_combined_" + dataset + ".png", dpi = 400)



if dataset == "truthfulqa_gen":
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Create two subplots in a single figure
    map = {0:[0,0], 1:[0,1], 2:[1,0], 3:[1,1]}
    for j, (metric_r, metric_nr) in enumerate(zip(metric_values, metric_values_norecomp)):
    
        ax = axs[*map[j]]

        if j == 0:
            metric_legend = "BLEU"
        elif j == 1:
            metric_legend = "ROUGE-1"
        elif j == 2:
            metric_legend = "ROUGE-2"
        elif j == 3:
            metric_legend = "ROUGE-L"        
        
        # Normalize the colors array for proper color mapping
        norm = plt.Normalize(vmin=min(range_th_norecomp), vmax=max(range_th_norecomp))
        cmap = plt.get_cmap('winter')
        
        # Plot the line for recomputation and no recomputation
        ax.plot(speedups_recomp, metric_r, color='#0BA4FF', zorder=1, label="Early exits Recomp")
        ax.plot(speedups_norecomp, metric_nr, color='#AEDD00', zorder=1, label="Early exits No Recomp")

        # Plot the points with colors
        scatter = ax.scatter(speedups_recomp, metric_r, c=range_th_recomp, cmap=cmap, norm=norm, marker='D')
        ax.scatter(speedups_norecomp, metric_nr, c=range_th_norecomp, cmap=cmap, norm=norm, marker='D')

        # Add the baseline as scatter points
        if baseline:
            ax.scatter(n_layers / (n_layers - torch.tensor(layers_dropped_baseline)), bl_values[j], 
                       color='#FF8C8C', zorder=2, label="Layer Pruning")

        ax.set_xlabel('Computational reduction factor')
        ax.set_ylabel(metric_legend)
        ax.grid()
        ax.set_title(f"Speedup vs {metric_legend}")

        # Add individual legends back to each subplot
        ax.legend()

    # Adjust layout to ensure space for colorbar and prevent overlapping
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust to leave space for colorbar on the right

    # Create a colorbar axis to the right of all subplots
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Threshold Value')  # Add a label to the colorbar

    # Save the combined subfigure
    plt.savefig(f"{path_weights_EE}/results/" + dataset + "/final_"+submetric+"_" + model_name + "_speedup_vs_combined_" + dataset + ".png", dpi=400)