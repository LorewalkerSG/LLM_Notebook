# In[0]
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["DEEPSEEK_API_KEY"] = ""



# In[1]
# tokenizer model 相匹配  tokenizer output -> model input'
# autoTokenizer autoModel:   Generic type

from transformers import AutoTokenizer, AutoModelForSequenceClassification


test_sentences = ['today is not that bad','today is so bad','so good']
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# In[]
batch_input = tokenizer(test_sentences,truncation=True,max_length=256,padding=True,return_tensors='pt')
# tokenizer.__call__ :encode
print(tokenizer.encode(test_sentences[0],))
print(tokenizer.tokenize(test_sentences[0],))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(test_sentences[0],)))
print(tokenizer.decode([ 101, 2651, 2003, 2025, 2008, 2919,  102]))
print(tokenizer.vocab)
print(tokenizer.vocab["[CLS]"])

print(tokenizer.special_tokens_map.values())
# dict_values(['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'])
# [100, 102, 0, 101, 103]
print(tokenizer.convert_tokens_to_ids([special  for special in tokenizer.special_tokens_map.values()]))


# In[]

batch_input = tokenizer(test_sentences,truncation=True,max_length=256,padding=True,return_tensors='pt')
# "padding=True" pad到最长
print(batch_input)

# batch_input2 = tokenizer(test_sentences,truncation=True,max_length=256,padding="max_length",return_tensors='pt')
# print(batch_input2)


# In[]
# 调用模型
import torch
import torch.nn.functional as F
# print(model.config)
with torch.no_grad():
    outputs = model(**batch_input)
    print(outputs)
    scores = F.softmax(outputs.logits,dim = -1)
    print(scores)
    labels = torch.argmax(scores,dim = -1)
    print(labels)
    cls_ = [model.config.id2label[id_] for id_ in labels.tolist()]
    print(cls_)
# %%
