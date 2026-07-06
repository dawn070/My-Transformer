import torch
import torch.nn as nn
import math

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head): # n_head 为注意力头数
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model

        d_k = d_model // n_head
        self.d_k = d_k
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):  # 如果是自注意力机制，那么 q=k=v
        batch, time, dimension = q.shape
        # 进行线性映射，权重互不共享
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 拆分成输入多个头的维数，比如 [128, 32, 512]→[128, 32, 8, 64]
        q = q.view(batch, time, self.n_head, self.d_k).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, self.d_k).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, self.d_k).permute(0, 2, 1, 3)

        score = q@k.transpose(2,3)/math.sqrt(self.d_k)

        if mask is not None:
            # 掩码为 0 的位置填充很小的数，以便Softmax后接近于0
            score = score.masked_fill(mask==0, -10000)
        
        score = self.softmax(score)@v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)
        output = self.w_combine(score)
        return output
    

# x = torch.rand(128, 32, 512)
# d_model = 512
# n_head = 8
# attention = MultiHeadAttention(d_model, n_head)

# output = attention(x, x, x)