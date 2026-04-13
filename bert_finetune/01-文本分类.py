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
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 200


#In[]
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
from datasets import load_dataset
emotions = load_dataset('emotion')
print(emotions['train']['text'], emotions['train']['label'])
print(emotions['train'].features)

labels = emotions['train'].features['label'].names
num_classes = len(labels)

def int2string(x):
    return labels[x]

# In[]
emotions_df = pd.DataFrame.from_dict(emotions['train'])
emotions_df[:5]

emotions_df['label_name'] = emotions_df['label'].apply(lambda x: labels[x])
emotions_df.label.value_counts()

plt.figure(figsize=(4, 3))
emotions_df['label_name'].value_counts(ascending=True).plot.barh()
plt.title('freq of labels')


plt.figure(figsize=(4,3))
emotions_df['words pre tweet'] = emotions_df['text'].str.split().apply(len)

# In[]
emotions_df.boxplot('words pre tweet',by ='label_name',
                    showfliers = False,
                    grid = False,
                    color = 'black')
plt.suptitle('')
plt.xlabel('')



# In[]

from transformers import AutoTokenizer
model_ckpt = 'distilbert-base-uncased' #uncased :不区分大小写
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


tokenizer.encode(emotions_df.iloc[6322]['text'])

print(tokenizer.vocab_size,tokenizer.model_max_length,tokenizer.model_input_names)
def batch_tokenize(batch):
    return tokenizer(batch['text'],padding = True,truncation = True)

emotions_encoded = emotions.map(batch_tokenize,batched = True,batch_size = None)

print(emotions_encoded)
print(emotions_encoded['train']['input_ids'])





emotions_encoded.set_format('torch',columns = ['label','input_ids','attention_mask'])
print(emotions_encoded['train']['input_ids'])


# In[]
# Fine tune
from transformers import AutoModelForSequenceClassification

model_ckpt = 'distilbert-base-uncased'


device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt,num_labels = num_classes).to(device)
print(model)

'''
部分权重没有被训练好
['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
DistilBertForSequenceClassification(
  (distilbert): DistilBertModel(
    (embeddings): Embeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (transformer): Transformer(
      (layer): ModuleList(
        (0-5): 6 x TransformerBlock(
          (attention): DistilBertSdpaAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (q_lin): Linear(in_features=768, out_features=768, bias=True)
            (k_lin): Linear(in_features=768, out_features=768, bias=True)
            (v_lin): Linear(in_features=768, out_features=768, bias=True)
            (out_lin): Linear(in_features=768, out_features=768, bias=True)
          )
          (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (ffn): FFN(
            (dropout): Dropout(p=0.1, inplace=False)
            (lin1): Linear(in_features=768, out_features=3072, bias=True)
            (lin2): Linear(in_features=3072, out_features=768, bias=True)
            (activation): GELUActivation()
          )
          (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
  )
  (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
  (classifier): Linear(in_features=768, out_features=6, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
)

'''
# In[]
from huggingface_hub import notebook_login
notebook_login()

# In[]
from transformers import TrainingArguments, Trainer


batch_size = 64
logging_steps = len(emotions_encoded['train'])// batch_size
model_name = f'{model_ckpt}_emotion_ft'

training_args = TrainingArguments(output_dir = model_name,
                                  num_train_epochs = 4,
                                  learning_rate = 2e-5,
                                  per_device_train_batch_size = batch_size,
                                  per_device_eval_batch_size = batch_size,
                                  weight_decay = 0.01,
                                  eval_strategy = 'epoch',
                                  disable_tqdm = False,
                                  push_to_hub = True,
                                  log_level='error')


# In[]

from transformers_utils import compute_classification_metrics
trainer = Trainer(model = model,
                  tokenizer = tokenizer,
                  train_dataset = emotions_encoded['train'],
                  eval_dataset = emotions_encoded['validation'],
                  args = training_args,
                  compute_metrics = compute_classification_metrics)


# In[]
trainer.train()


# In[]

pred_output = trainer.predict(emotions_encoded['validation'])

print(pred_output)

# In[]

y_pred = np.argmax(pred_output.predictions,axis = -1)
y_true = pred_output.label_ids


from transformers_utils import plot_confusion_matrix

plot_confusion_matrix(y_pred,y_true,labels)

# In[]
# Huggingface 上传和下载

trainer.push_to_hub(commit_message =  'Training Completed!')

from transformers import pipeline

model_id = "lorewalkerSG/distilbert-base-uncased_emotion_ft"
classifier = pipeline('text-classification',model = model_id)

custom_tweet = "I saw a mmovie today and it was really good"

preds = classifier(custom_tweet,return_all_scores = True)

preds_df  = pd.DataFrame(preds[0])
plt.bar(labels,100*preds_df['score'],color = 'C0')
plt.show()
a = ''

# %%
