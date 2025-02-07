import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Set a modern aesthetic for the plot
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
})

# Define the path to the folder where the files are located
path = f"./weights/mistral"
path_weights_EE_mistral = path + f"/EE_1_layers_middle_2_pos_15_19_23_27"
dataset = "mmlu"
submetric = "acc" # acc, diff, max
recomputation = True
baseline = True

path = "./weights/mamba"
path_weights_EE_mamba = path + f"/EE_1_layers_middle_2_pos_32_40_48_56"
dataset = "mmlu"
recomputation = False
baseline = True
recomp = "/recompute_states" if recomputation else "/no_recomp" 

# Load data from JSON files
with open(f"{path_weights_EE_mamba}/results/"+dataset+"/baseline/results_list.json", "r") as f:
    results_list_baseline_mamba = json.load(f)

with open(f"{path_weights_EE_mamba}/results/"+dataset+"/baseline/layers_dropped.json", "r") as f:
    layers_dropped_baseline_mamba = json.load(f)

with open(f"{path_weights_EE_mamba}/results/"+dataset+"/baseline/5_results_list.json", "r") as f:
    results_list_baseline_mamba_n5 = json.load(f)

with open(f"{path_weights_EE_mamba}/results/"+dataset+"/baseline/5_layers_dropped.json", "r") as f:
    layers_dropped_baseline_mamba_n5 = json.load(f)

with open(f"{path_weights_EE_mistral}/results/"+dataset+"/baseline/results_list.json", "r") as f:
    results_list_baseline_mistral = json.load(f)

with open(f"{path_weights_EE_mistral}/results/"+dataset+"/baseline/layers_dropped.json", "r") as f:
    layers_dropped_baseline_mistral = json.load(f)

with open(f"{path_weights_EE_mistral}/results/"+dataset+"/baseline/5_results_list.json", "r") as f:
    results_list_baseline_mistral_n5 = json.load(f)

with open(f"{path_weights_EE_mistral}/results/"+dataset+"/baseline/5_layers_dropped.json", "r") as f:
    layers_dropped_baseline_mistral_n5 = json.load(f)

# Prepare data for plotting
n_layers_mamba = 64
n_layers_mistral = 32
metrics = ["acc,none"]
metric_values = []
bl_values_mamba = []
bl_values_mamba_n5 = []
bl_values_mistral = []
bl_values_mistral_n5 = []
for i, metric in enumerate(metrics):
    bl_values_mamba.append([r["results"][dataset][metric] for r in results_list_baseline_mamba])
    bl_values_mamba_n5.append([r["results"][dataset][metric] for r in results_list_baseline_mamba_n5])
    bl_values_mistral.append([r["results"][dataset][metric] for r in results_list_baseline_mistral])
    bl_values_mistral_n5.append([r["results"][dataset][metric] for r in results_list_baseline_mistral_n5])

# Create the plot
plt.figure(figsize=(12, 6))

# Plot each line with enhanced styling
plt.plot([r/n_layers_mamba for r in layers_dropped_baseline_mamba], bl_values_mamba[0], 
         label="Mamba", linewidth=2.5, marker='o', markersize=6, color="#0000ffff")
plt.plot([r/n_layers_mamba for r in layers_dropped_baseline_mamba_n5], bl_values_mamba_n5[0], 
         label="Mamba n_shots=5", linewidth=2.5, linestyle='--', marker='s', markersize=6, color="#0072c6ff")
plt.plot([r/n_layers_mistral for r in layers_dropped_baseline_mistral], bl_values_mistral[0], 
         label="Transformer", linewidth=2.5, marker='^', markersize=6, color="#00a8abff")
plt.plot([r/n_layers_mistral for r in layers_dropped_baseline_mistral_n5], bl_values_mistral_n5[0], 
         label="Transformer n_shots=5", linewidth=2.5, linestyle='--', marker='D', markersize=6, color="#00ff80ff")

# Add baseline with enhanced styling
plt.axhline(y=0.25, color='r', linestyle='--', linewidth=2, label="Baseline")

# Labels and title
plt.xlabel("Ratio of Layers Dropped", weight='bold')
plt.ylabel("Accuracy", weight='bold')
plt.title("Accuracy vs Ratio of Layers Dropped (MMLU)", pad=20)
plt.legend(frameon=True, shadow=True, fontsize='medium', fancybox=True, loc='upper right')

# Set x-axis limit to cut at 0.7 layers dropped
plt.xlim(0, 0.7)

# Additional aesthetics
plt.grid(visible=True, linestyle=':', linewidth=0.7)
plt.tight_layout()

# Save the plot with higher DPI for clarity
plt.savefig(f"{path_weights_EE_mamba}/results/"+dataset+"/baseline"+f"/{metric}_vs_layers_dropped.png", dpi=500)
plt.savefig(f"{path_weights_EE_mistral}/results/"+dataset+"/baseline"+f"/{metric}_vs_layers_dropped.png", dpi=500)

# plt.show()
