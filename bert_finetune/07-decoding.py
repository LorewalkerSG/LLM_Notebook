# In[]

import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# In[]

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import numpy as np
device  = 'cuda' if torch.cuda.is_available() else 'cpu'


model_ckpt = 'gpt2-xl'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForCausalLM.from_pretrained(model_ckpt).to(device)


sample_text = 'A long long time ago, '

model_inputs = tokenizer(sample_text,return_tensors = 'pt')
print(model_inputs)

# In[]



input_ids = model_inputs['input_ids'].to(device)
print(input_ids)




n_steps = 10
choices_per_step = 5

iterations = []
with torch.no_grad():
    for _ in range(n_steps):
        iteration = {}
        iteration['input'] = tokenizer.decode(input_ids[0])
        output = model(input_ids = input_ids)

        last_token_logits = output.logits[0,-1,:]
        last_token_probs = torch.softmax(last_token_logits,dim = -1)
        sorted_ids = torch.argsort(last_token_probs,dim = -1,descending = True)

        for choice_idx in range(choices_per_step):
            token_id = sorted_ids[choice_idx]
            token_prob = last_token_probs[token_id].cpu().numpy()
            token_choice = f'{tokenizer.decode(token_id)}({100*token_prob:.2f}%)'
            iteration[f'choice {choice_idx +1 }'] = token_choice

        print('before append input_ids.shape', input_ids.shape)
        input_ids = torch.cat([input_ids,sorted_ids[None,0,None]],dim = -1)
        print("after append input_ids.shape",input_ids.shape)
        iterations.append(iteration)


# In[]

import pandas  as pd 
df = pd.DataFrame(iterations)
print(iterations[-1])
print(df)

# In[]
input_ids = tokenizer(sample_text,return_tensor = 'pt').to(device)





# %%
