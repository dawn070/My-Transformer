import torch
from torch import nn
import torch.nn.functional as F
import math

# random_mat = torch.rand(4,4)
# print(random_mat)

# 定义词嵌入类
# 将输入的词汇表转换为指定维度的向量
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model): # vocab_sizes是词表大小，d_model是模型维度（输出向量的长度）
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)  # 调用父类构造函数

# 位置编码
class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionEmbedding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)  # 创建全0矩阵
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)  # 转换为二维张量
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:,0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2] = torch.cos(pos/(10000**(_2i/d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()  # 获取长度和大小
        return self.encoding[:seq_len,:]
    
# 最终编码
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()

        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)