import torch
from torch import nn
import torch.nn.functional as F
import math

from embedding import TransformerEmbedding
from attention import MultiHeadAttention
from layernorm import LayerNorm
from encoder import Encoder
from decoder import Decoder

# 变压器类
class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx,
                 enc_voc_size, dec_voc_size, max_len,
                 d_model, n_head, ffn_hidden, n_layer,
                 drop_prob=0.1, device='cuda'):
        super(Transformer, self).__init__()

        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_head, 
                 n_layer, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_head, 
                 n_layer, drop_prob, device)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # 设置位置填充掩码
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)  # 先判断后匹配
        q = q.repeat(1, 1, 1, len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(3)  # 先判断后匹配
        k = k.repeat(1, 1, len_q, 1)
        mask = q&k
        return mask
    
    # 设置因果掩码，防止未来信息泄露
    def make_causal_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask
    
    def forward(self, src, trg):
        # 创造填充掩码
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) & self.make_causal_mask(trg, trg)
        # 编码-译码
        enc = self.encoder(src, src_mask)
        out = self.decoder(trg, enc, trg_mask, src_mask)

        return out