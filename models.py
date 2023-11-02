import os.path as osp
import numpy as np
import torch
from torch.nn import ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import negative_sampling
import random

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for i in range(0, num_layers-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        hidden = z[edge_label_index[0]] * z[edge_label_index[1]]
        logits = (hidden).sum(dim=-1)
        hidden = F.normalize(hidden, dim=1)
        return hidden, logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for i in range(0, num_layers-2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        hidden = z[edge_label_index[0]] * z[edge_label_index[1]]
        logits = (hidden).sum(dim=-1)
        hidden = F.normalize(hidden, dim=1)
        return hidden, logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=8, att_dropout=0):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels//heads, heads=heads, dropout=att_dropout))
        for i in range(0, num_layers-2):
            self.convs.append(GATConv(hidden_channels, hidden_channels//heads, heads=heads, dropout=att_dropout))
        self.convs.append(GATConv(hidden_channels, out_channels, dropout=att_dropout))

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        hidden = z[edge_label_index[0]] * z[edge_label_index[1]]
        logits = (hidden).sum(dim=-1)
        hidden = F.normalize(hidden, dim=1)
        return hidden, logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.layers = ModuleList()
        self.layers.append(torch.nn.Linear(in_channels, hidden_channels))
        for i in range(0, num_layers-2):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(torch.nn.Linear(hidden_channels, out_channels))

    def encode(self, x, edge_index):
        for fc in self.layers[:-1]:
            x = fc(x).relu()
        x = self.layers[-1](x)
        return x

    def decode(self, z, edge_label_index):
        hidden = z[edge_label_index[0]] * z[edge_label_index[1]]
        logits = (hidden).sum(dim=-1)
        hidden = F.normalize(hidden, dim=1)
        return hidden, logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()