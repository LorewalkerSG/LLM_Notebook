# In[]
import torch
import torch.nn as nn
import torch.nn.functional as F
logits = torch.randn(2,3,4) # 两句话 ，每句话长度是3， 使用四位表示一个长度
label = torch.randint(0,4,(2,3))

logits = logits.transpose(1,2)
# 六个三次平均的交叉熵
# reduction= none 返回所有单词的交叉熵
F.cross_entropy(logits,label)
tgt_len = torch.Tensor([2,3]).to(torch.int32)
mask = torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(tgt_len) - L)),0) for L in tgt_len])
# 手动添加掩码，在求平均
F.cross_entropy(logits,label,reduction='none') * mask
# 设置ignore_index ,值为ignore index 的不会贡献梯度
label[0,2] = -100

print(F.cross_entropy(logits,label,reduction='none'))






# %%
