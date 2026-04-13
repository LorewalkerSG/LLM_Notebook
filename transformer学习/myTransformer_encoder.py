# In[1]
import torch
import numpy 
import torch.nn as nn
import torch.nn.functional as F


batch_size = 2

# 单词表大小
max_num_src_words = 8
max_num_tgt_words = 8
model_dim = 8

# src_len = torch.randint(2,5,(batch_size,))
# tgt_len = torch.randint(2,5,(batch_size,))
src_len = torch.Tensor([2,4]).to(torch.int32)
tgt_len = torch.Tensor([4,3]).to(torch.int32)
max_src_seq_len = 5
max_tgt_seq_len = 5
max_position_len = 5


# 单词索引构成的句子
src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_src_words,(L,)),(0,max_src_seq_len  - L)),0) for L in src_len])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_tgt_words,(L,)),(0,max_tgt_seq_len - L)),0) for L in tgt_len])



# 构造embeddng
src_embedding_table =  nn.Embedding(max_num_src_words+1,model_dim)  # 考虑pad加入了0
tgt_embedding_table =  nn.Embedding(max_num_tgt_words+1,model_dim)  

src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)


# 构造pos emb



pos_mat = torch.arange(max_position_len).reshape((-1,1))
i_mat = torch.pow(10000,torch.arange(0,model_dim,2).reshape((1,-1))/model_dim)
pe_embedding_table = torch.zeros(max_position_len,model_dim)

pe_embedding_table[:,0::2] = torch.sin(pos_mat/i_mat)
pe_embedding_table[:,1::2] = torch.cos(pos_mat/i_mat)


pe_embedding = nn.Embedding(max_position_len,model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding_table,requires_grad=False)


src_pos = torch.arange(max(src_len)).repeat(len(src_len),1).to(torch.int32)
tgt_pos = torch.arange(max(tgt_len)).repeat(len(tgt_len),1).to(torch.int32)

# src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)),0) for _ in src_len]).to(torch.int32)
# tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)),0) for _ in tgt_len]).to(torch.int32)
src_pe_embdding = pe_embedding(src_pos)
tgt_pe_embdding = pe_embedding(tgt_pos)

# In[2]
# softmax演示  scale 的重要性
# alpha1 = 0.1
# alpha2 = 10
# score = torch.randn(5)

# def softmax_func(score):
#     return F.softmax(score)
# jaco_mat1 = torch.autograd.functional.jacobian(softmax_func,score * alpha1)
# jaco_mat2 = torch.autograd.functional.jacobian(softmax_func,score * alpha2)

# print(jaco_mat1,jaco_mat2)


# prob = F.softmax(score * alpha1, -1)

# print(score)
# print(prob)



# In[]
# 构造encoder的self-sttention mask 
# mask的shape：[batch_size, max_src_len, max_src_len], 值为1 或inf


valid_encoder_pos = torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(src_len)-L)),0) for L in src_len ]).unsqueeze(2)
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos,valid_encoder_pos.transpose(1,2))

score = torch.randn(batch_size,max(src_len),max(src_len))
invalid_encoder_pos_matrix= 1-valid_encoder_pos_matrix
masked_encoder_self_attention  = invalid_encoder_pos_matrix.to(torch.bool)
masked_score = score.masked_fill(masked_encoder_self_attention,-1e9)
prob = F.softmax(masked_score,-1)





# In[]
# step 5 : 构造intra-attention 的mask
# Q @ K^T shape:[batch_size tgt_seq_len src_seq_len ]



valid_encoder_pos = torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(src_len)-L)),0) for L in src_len ]).unsqueeze(2)
valid_decoder_pos = torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(tgt_len)-L)),0) for L in tgt_len ]).unsqueeze(2)

valid_cross_pos_matrix = torch.bmm(valid_decoder_pos,valid_encoder_pos.transpose(1,2))
invalid_cross_pos_matrix = 1- valid_cross_pos_matrix
mask_cross_attention = invalid_cross_pos_matrix.to(torch.bool)
print(mask_cross_attention)


# In[]
# step 6 构造decoder self-attention 的mask
tri_matrix = [torch.tril(torch.ones((L,L))) for L in tgt_len]
# 给一个特殊字符，让解码器预测下一个字符，再将所有输出结果送给模型作为输入，以此类推


# 在 PyTorch 中，F.pad 对于 2D 矩阵的参数顺序是 (左, 右, 上, 下)
valid_decoder_tri_matrix = torch.cat([torch.unsqueeze(F.pad(torch.tril(torch.ones((L,L))),(0,max(tgt_len)-L,0,max(tgt_len)-L)),0) for L in tgt_len])
invalid_decoder_tri_matrix = 1- valid_decoder_tri_matrix
invalid_decoder_tri_matrix = valid_decoder_tri_matrix.to(torch.bool)
score = torch.randn(batch_size,max(tgt_len),max(tgt_len))
masked_score = score.masked_fill(invalid_decoder_tri_matrix,-1e-9)
prob = F.softmax(masked_score,-1)



# step 7 构建scaled self-attention
def scaled_dot_product_attention(Q,K,V,attn_mask):
    # shape of Q K V : batch_size * num_head , seq_len, model_dim/num_head
    # 这是最关键的一步。假设模型总维度 model_dim ($d_{model}$) 是 512，共有 8 个头（num_head）：
    # 物理含义：我们不是让一个 512 维的注意力机制去处理所有信息，而是把它切成 8 个 64 维的小块。为什么要切分？ 
    # 就像一个团队，如果所有人（512人）都在讨论同一个问题，效率可能不高；如果分成 8 个小组（每组 64 人），
    # 第一组可以专门研究“语法”，第二组研究“语义”，第三组研究“指代关系”，最后再汇总。
    score = torch.bmm(Q, K.transpose(-1,-2))/torch.sqrt(model_dim)
    masked_score = score.masked_fill(attn_mask,-1e9)
    prob = F.softmax(masked_score,-1)
    context =  torch.bmm(prob,V)
    return context








# target 是Q src是K


# valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(src_len)-L)),0) for L in src_len]),2)
# valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos,valid_encoder_pos.transpose(1,2))
# print(valid_encoder_pos_matrix.shape)



# %%
# 三种mask：
# encoder 输入/输出序列是否被pad
# cross： 输出和哪些输入有关
# decoder： 因果