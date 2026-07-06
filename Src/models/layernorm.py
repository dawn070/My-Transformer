import torch
import torch.nn as nn

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()

        # nn.Parameter 设置为可学习参数
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased = False, keepdim=True)
        # 归一化
        output = (x-mean)/torch.sqrt(var+self.eps)
        # 缩放 + 平移 : 使网络有不同的分布特征
        output = self.gamma * output + self.beta
        return output