# In[]

import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"







# In[]
from transformers import BertTokenizer,BertModel

import torch
from torch import nn

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

test_sentence = 'this is a test sentence'
input = tokenizer(test_sentence,return_tensors = 'pt')

input_ids = input['input_ids']
print("input_ids.shape",input_ids.shape)
token_type_ids = input['token_type_ids']

_ = model.embeddings(input_ids,token_type_ids) 
print(_)


pos_ids = torch.arange(input_ids.shape[-1])



# In[]
# word embedding
word_embeddings = model.embeddings.word_embeddings(input_ids)
# segment_embeddings
segment_embeddings = model.embeddings.token_type_embeddings(token_type_ids)

# position embeddings 
position_embeddings = model.embeddings.position_embeddings(pos_ids)

embeddings = word_embeddings + segment_embeddings+  position_embeddings
print(embeddings)





# %%
