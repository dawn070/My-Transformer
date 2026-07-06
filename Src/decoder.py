import torch
from torch import nn

from embedding import TransformerEmbedding
from attention import MultiHeadAttention
from layernorm import LayerNorm
from encoder import PositionwiseFeedForward

# 译码层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob=0.1):
        super(DecoderLayer, self).__init__()

        self.attention1 = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # 交叉注意力
        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        x0 = dec

        x = self.attention1(dec, dec, dec, t_mask)
        x = self.dropout1(x)
        x = self.norm1(x+x0)
        x0 = x

        x = self.cross_attention(x, enc, enc, s_mask)
        x = self.dropout2(x)
        x = self.norm2(x+x0)
        x0 = x

        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x+x0)

        return x
    
# 译码器
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden,
                  n_head, n_layer, drop_prob=0.1, device='cuda'):
        super(Decoder, self).__init__()

        self.embedding = TransformerEmbedding(
            vocab_size=dec_voc_size,
            max_len=max_len,
            d_model=d_model,
            drop_prob=drop_prob,
            device=device
        )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layer)  # 多个Encoder层
            ]
        )
        # 映射回词汇表维度数
        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(enc)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        dec = self.fc(dec)
        
        return dec