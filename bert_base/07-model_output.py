# In[]
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"



# In[]
# wordpiece 将词分成子词
from transformers import BertTokenizer,BertModel
import torch
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name,output_hidden_states = True)

text = "After stealing money from the bank vault, the bank robber was seen "\
    "fishing on the Mississippi river bank."

token_input = tokenizer(text,return_tensors = 'pt')
# token_input = tokenizer(text)


# print(tokenizer.convert_ids_to_tokens(token_input["input_ids"]))
# 无视大写


print(token_input)
print(token_input["input_ids"].shape)

# In[]
# forward : embedding -> encoder ->pooling

# outputs[1]： pooler的输出
# len(outputs[2])= 13 : embedding输入和12层的输出 
# model.embeddings(token_input['input_ids'],token_input['token_type_ids']) == outputs[2][0]

model.eval()
with torch.no_grad():
    outputs = model(**token_input)


# outputs[2][-1] == outputs[0]
# outputs[1] = model.pooler(outputs[2][-1]) 
# 模型的pooler 会取出第一个token


# %%
