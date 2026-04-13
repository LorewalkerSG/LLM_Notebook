# In[]
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

#In[]

from transformers import BertTokenizer
import torch 
from torch import nn
import math
from bertviz.transformers_neuron_view import BertModel,BertConfig

max_length = 256
model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(model_name,output_attentions = True,
                                    output_hidden_states = True,
                                    return_dict = True)
tokenizer = BertTokenizer.from_pretrained(model_name)

config.max_posiiton_embeddings = max_length
model = BertModel(config).from_pretrained(model_name)
model.eval()




# %%
