import torch

from utils import get_model_and_tok_with_EE

model_choice = "mistral" # gpt2, mistral, mamba

tokens_generated = 50

size = "350" # 124M, 350M, 774M, 1558M
# path = f"./weights/gpt2/gpt2_{size}M_100B_FinewebEdu_hf"

path = f"./weights/mamba"
# path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_32_40_48_56"
path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_31_39_47_55"

# path = f"./weights/mistral"
# # path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_16_20_24_28"
# path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_15_19_23_27"
plot_intermediate_states = True
th_for_EE = 0.6
ee_pos = [int(p) for p in path_weigths_EE.split("_pos_")[-1].split("_")]
recompute_states = False

model, encode, decode = get_model_and_tok_with_EE(model_choice, path, path_weigths_EE, ee_pos, th_for_EE)

model.th = torch.tensor([0.4,0.3,0.2,0.2])*2

# inputs = "What can you tell me about flowers?"
inputs = "Transformer models are right now one of the most popular"

print("\n \t No EEss\n")

with torch.no_grad():
    output1 = model.generate(encode(inputs).to("cuda"), temperature=1.5, max_new_tokens=tokens_generated, top_k = 10, use_EE = False)

print(decode(output1))

print("\n \t With EE\n")  
if model_choice == "mamba":
    model.refresh_generation()
    
with torch.no_grad():
    output2 = model.generate(encode(inputs).to("cuda"), temperature=1.5, max_new_tokens=tokens_generated, top_k = 10, use_EE = True, recompute_states=recompute_states)

output_text = decode(output2)

exits_done = model.exits_done
positions_exit = model.positions_exit

# capitalize words in exits
output_text = " ".join([word.upper() if i in positions_exit else word for i, word in enumerate(output_text.split())])
print(output_text)

# sum 1 if we use last block
if model_choice == "mistral":
    n_layer = 32
    print(exits_done)
    saved = sum([n_layer - e -1 for e in exits_done])
elif model_choice == "mamba":
    n_layer = 64
    saved = sum([n_layer - e for e in exits_done])

print(f"EEs saved {100*saved/(n_layer*tokens_generated)}% computation")
