import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from layernorm import LayerNorm
from embedding import TransformerEmbedding

# 前馈神经网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x= self.fc2(x)
        return x
    
# 编码层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask=None):
        x0 = x  # 保存原始的输入

        x= self.attention(x, x, x, mask) # q=k=v=x
        x = self.dropout1(x)
        x = x + x0  # 残差连接
        x = self.norm1(x)
        x0 = x  # 保存此时x值

        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + x0
        x = self.norm2(x)

        return x
    
# 编码器
class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, 
                 n_layer, drop_prob=0.1, device='cuda'):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(
            vocab_size=enc_voc_size,
            max_len=max_len,
            d_model=d_model,
            drop_prob=drop_prob,
            device=device
        )
        
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layer)  # 多个Encoder层
            ]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x= layer(x, s_mask)  # 上一层输出作为下一层输入，逐层提取特征

        return x