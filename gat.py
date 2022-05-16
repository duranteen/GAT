import torch
from torch import nn
from dgl.nn import GATConv


class GAT(nn.Module):
    def __init__(self, graph, num_layers,
                 input_dim, hidden_dim,
                 num_classes, heads, activation,
                 feat_drop, attn_drop,
                 negative_slope, residual):
        """

        :param graph: 图
        :param num_layers: 网络层数
        :param input_dim: 输入特征维度
        :param hidden_dim: 输出特征维度
        :param num_classes: 分类数
        :param heads: 每层多头注意力头数
        :param activation: 激活函数
        :param feat_drop: 特征的dropout rate
        :param attn_drop: 注意力权重的dropout rate
        :param negative_slope: LeakyReLU
        :param residual: 是否使用残差连接
        """
        super(GAT, self).__init__()
        self.graph = graph
        self.num_layers = num_layers
        self.activation = activation
        self.gat_layers = nn.ModuleList()
        if num_layers > 1:
            self.gat_layers.append(GATConv(input_dim, hidden_dim,
                                           heads[0], feat_drop, attn_drop,
                                           negative_slope, False, self.activation))
            for l in range(1, self.num_layers-1):
                self.gat_layers.append(GATConv(hidden_dim*heads[l-1], hidden_dim, heads[l],
                                               feat_drop, attn_drop, negative_slope,
                                               residual, self.activation))
            self.gat_layers.append(GATConv(hidden_dim*heads[-2], num_classes, heads[-1],
                                           feat_drop, attn_drop, negative_slope,
                                           residual, None))
        else:
            self.gat_layers.append(GATConv(input_dim, num_classes, heads[0],
                                           feat_drop, attn_drop, negative_slope,
                                           residual, None))

    def forward(self, x):
        h = x
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.graph, h)
            h = h.flatten(1) if l != self.num_layers-1 else h.mean(1)
        return h
