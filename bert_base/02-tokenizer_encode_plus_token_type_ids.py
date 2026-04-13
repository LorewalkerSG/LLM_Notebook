
# In[]
from transformers import BertTokenizer 
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["DEEPSEEK_API_KEY"] = ""


model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name)
print(tokenizer)


# In[]
# 文本语料
# newsgroups_train.DESCR
# newsgroups_train.data
# newsgroups_train.target
# newsgroups_train.target_names

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')


# print(newsgroups_train.DESCR)
# print(newsgroups_train.data)
# print(newsgroups_train.target)
# print(newsgroups_train.target_names)

from collections import Counter
Counter(newsgroups_train.target)
test_news = newsgroups_train.data[:3]

# 单句级别
tokenizer(test_news,truncation = True,max_length = 32)
list(tokenizer(test_news,truncation = True,max_length = 32).keys())
# ['input_ids', 'token_type_ids', 'attention_mask']

# sentence pair 级别
# 用于nsp (next sentence predict 下个句子预测)(bert 预训练任务)
tokenizer.encode_plus(text = test_news[0],text_pair = test_news[1],max_length = 32,truncation = True)
# token_type_ids 0 表示第一句 1 表示第二句
# 可以通过tokenizer() (tokenizer.__call__); 也可以通过encoder_plus 生成，前一句为0， 后一句为1

tokenizer.decode([101, 2013, 1024, 3393, 2099, 2595, 3367, 1030, 11333, 2213, 1012, 8529, 2094, 1012, 3968, 2226, 102, 2013, 1024, 3124, 5283, 2080, 1030, 9806, 1012, 1057, 1012, 2899, 1012, 3968, 2226, 102])




# attention mask bert训练时用的，不添加掩码就全是1




# %%
