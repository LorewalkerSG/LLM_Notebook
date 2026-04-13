# In[]


import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["DEEPSEEK_API_KEY"] = ""


model_name = "bert-base-uncased"



# In[]
from transformers import BertModel,BertForSequenceClassification

model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
cls_model = BertForSequenceClassification.from_pretrained(model_name)
print(model)
print("~"*20)
print(cls_model)

# (word_embeddings): Embedding(30522, 768, padding_idx=0)
    #   (position_embeddings): Embedding(512, 768)
    #   (token_type_embeddings): Embedding(2, 768)    用于nsp任务



# In[]
# 参数量统计
total_param = 0
total_learnable_parameters =0 
for name,param in model.named_parameters():
    print(name,"->",param.shape,"->" ,param.numel()) # .numel() number of elements
    if param.requires_grad:
        total_learnable_parameters += param.numel()
    total_param+=param.numel()
print(total_learnable_parameters)
print(total_param)
# %%
