# In[]

import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"



# In[]
import torch
from torch import nn
import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

def scaled_dot_production_attention(query,key,value):
    dim_k = key.size(-1)
    # atten_scores = torch.bmm(query,key)/np.sqrt(dim_k)
    attened_weight  = torch.bmm(F.softmax(torch.bmm(query,key.transpose(1,2))/np.sqrt(dim_k),dim = -1),value)
    return attened_weight


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.Wq = nn.Linear(embed_dim,head_dim)
        self.Wk = nn.Linear(embed_dim,head_dim)
        self.Wv = nn.Linear(embed_dim,head_dim)

    def forward(self,hidden_states):
        q = self.Wq(hidden_states)
        k = self.Wk(hidden_states)
        v = self.Wv(hidden_states)
        return scaled_dot_production_attention(q,k,v)
    
class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim//num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim,head_dim)for _ in range(num_heads)])
        self.output_layer = nn.Linear(embed_dim,embed_dim)
    def forward(self,hidden_states):
        x = torch.cat([head(hidden_states)for head in self.heads] ,dim = -1)
        x = self.output_layer(x)
        return x
    

# In[]
from transformers import AutoConfig, AutoTokenizer, AutoModel
model_ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained(model_ckpt)
print(config)

# In[]

token_embedding  = nn.Embedding(config.vocab_size,config.hidden_size)
sample_text = 'time flies like an arrow'
model_inputs = tokenizer(sample_text,return_tensors = 'pt',add_special_tokens = False)
input_embeddings = token_embedding(model_inputs['input_ids'])
print(input_embeddings)
input_embeddings.shape # 批次1 长度5 embed_dim 768
# In[]
mha = MultiHeadAttention(config)
attn_output = mha(input_embeddings)
print(mha(input_embeddings))


# In[]

'''
intermediate_size = 4 * hidden_size
GELU: f(x) = x \phi(x)  (标准正态分布的分布函数)
'''
# x = np.arange(-5,5,0.01)
# plt.plot(x,nn.GELU()(torch.from_numpy(x)))
# plt.show()

class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size,config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gelu = nn.GELU()
    def forward(self,x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
ffn = FeedForward(config)

print(ffn(attn_output).size())

# In[]
class TransformerEncoderLayer(nn.Module):
    # 前置归一化
    def __init__(self,config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    

# %%
