import sys
import os
sys.path.append(os.path.join(sys.path[0], '../../EleutherAI_Eval_harness'))
sys.path.append(os.path.join(sys.path[0], '../../'))

# from lm_eval.models.mistral_models import Mistral_7b
from lm_eval.models.gpt2_custom import gpt2_custom

from lm_eval import evaluate, simple_evaluate
from lm_eval.tasks import get_task_dict
from lm_eval.utils import make_table

# os.chdir(os.path.join(sys.path[0], './EE_Clean'))

from models.gpt2.model import GPT, GPTConfig
# import transformers
import tiktoken

import json
import torch

size = "350" # 124M, 350M, 774M, 1558M
path = f"./weights/gpt2/gpt2_{size}M_100B_FinewebEdu_hf"
max_length = 1024
max_gen_tokens = 64
device = "cuda"
batch_size = 1

# create your model (could be running finetuning with some custom modeling code)
print("Loading model...")

# open config file
with open(path + "/config.json") as f:
    config = json.load(f)

# dump config into GPTConfig
config_dataclass = GPTConfig(   block_size = config['n_ctx'],
                                vocab_size = config['vocab_size'],
                                n_layer = config['n_layer'],
                                n_head = config['n_head'],
                                n_embd = config['n_embd'],
                                dropout = config['attn_pdrop'],
                                bias = True,
                            )

model = GPT(config_dataclass)
model.from_hf(path + "/model.safetensors")

print("Loading tokenizer...")

# load hf tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
# tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file = path + "/tokenizer_config.json")

# args.max_batch_size = batch_size


n_layer = model.config.n_layer

model.eval()
model.to(device = device, dtype = torch.float)


# instantiate an LM subclass that takes your initialized model and can run
# - `Your_LM.loglikelihood()`
# - `Your_LM.loglikelihood_rolling()`
# - `Your_LM.generate_until()`
lm_obj = gpt2_custom(model=model, tokenizer = tokenizer, batch_size=batch_size, max_length = max_length, max_gen_tokens = max_gen_tokens,device = device)

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

results = simple_evaluate(
    model = lm_obj,
    tasks = ["truthfulqa"],
    num_fewshot = 0,
)

print(make_table(results))