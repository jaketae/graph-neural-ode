import math
from typing import Callable

import dgl
import dgl.function as fn
import torch
from torch import nn


class GCNLayer(nn.Module):
    def __init__(
        self,
        g: dgl.DGLGraph,
        in_feats: int,
        out_feats: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        dropout: int,
        bias: bool = True,
    ):
        super().__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.0
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * self.g.ndata["norm"]
        self.g.ndata["h"] = h
        self.g.update_all(fn.copy_u("h", "m"), fn.sum(msg="m", out="h"))
        h = self.g.ndata.pop("h")
        # normalization by square root of dst degree
        h = h * self.g.ndata["norm"]
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h


class GCN(nn.Module):
    def __init__(
        self,
        num_layers: int,
        g: dgl.DGLGraph,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        activation: Callable,
        dropout: int,
        bias=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(g, in_feats, hidden_feats, activation, dropout))

        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(g, hidden_feats, hidden_feats, activation, dropout))

        self.layers.append(GCNLayer(g, hidden_feats, out_feats, None, 0.0))

    def set_graph(self, g):
        for layer in self.layers:
            layer.g = g

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h
