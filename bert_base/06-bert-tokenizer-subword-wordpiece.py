# In[]
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"



# In[]
# wordpiece 将词分成子词
from transformers import BertTokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

'''
print(tokenizer.vocab)# ('[PAD]', 0)
print(tokenizer.ids_to_tokens) # (0, '[PAD]'),
'''


# s1 = 'album sold 500,000 copies'

s1 = 'album sold 124443286539 copies'
s2 = 'technically perfect, melodically correct'
s3 = 'featuring a previously unheard track'
s4 = 'best-selling music artist'
s5 = 's1 d1 o1 and o2'
s6 = 'asdofwheohwbeif'


# In[]

inputs = tokenizer(s6)
print(inputs)

print(tokenizer.convert_ids_to_tokens(inputs['input_ids']))



# In[]
cnt = 0
for word in tokenizer.vocab:
    if word.startswith('##') :
        cnt += 1
        print(word)
print(cnt)
# %%
