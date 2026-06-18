import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(nn.Module):
    """Standard GCN — supports both node classification and graph classification.

    Set pool=True for graph-level tasks (adds global mean pooling before classifier).
    """

    def __init__(self, input_dim: int, hid_dim: int, n_classes: int,
                 n_layers: int, dropout: float = 0.3, pool: bool = False):
        super().__init__()
        self.n_layers = n_layers
        self.dropout  = dropout
        self.pool     = pool
        self.layers   = nn.ModuleList()

        if n_layers == 0:
            self.layers.append(nn.Linear(input_dim, hid_dim))
        else:
            dims = [input_dim] + [hid_dim] * n_layers
            for i in range(n_layers):
                self.layers.append(GCNConv(dims[i], dims[i + 1]))

        self.classifier = nn.Linear(hid_dim, n_classes)

    def _embed(self, x, edge_index):
        if self.n_layers == 0:
            x = F.relu(self.layers[0](x))
            return F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.layers:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, batch=None):
        h = self._embed(x, edge_index)
        if self.pool:
            h = global_mean_pool(h, batch)
        return self.classifier(h)

    def embed(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            return self._embed(x, edge_index)


class SkipGCN(nn.Module):
    """GCN with residual (skip) connections — adapted from Practical 3.

    Each layer: H^(t) = ReLU(GCNConv(H^(t-1))) + proj(H^(t-1))
    Set pool=True for graph-level classification.
    """

    def __init__(self, input_dim: int, hid_dim: int, n_classes: int,
                 n_layers: int, dropout: float = 0.3, pool: bool = False):
        super().__init__()
        self.n_layers  = n_layers
        self.dropout   = dropout
        self.pool      = pool
        self.layers    = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        if n_layers == 0:
            self.layers.append(nn.Linear(input_dim, hid_dim))
        else:
            dims = [input_dim] + [hid_dim] * n_layers
            for i in range(n_layers):
                self.layers.append(GCNConv(dims[i], dims[i + 1]))
                if dims[i] != dims[i + 1]:
                    self.res_projs.append(nn.Linear(dims[i], dims[i + 1], bias=False))
                else:
                    self.res_projs.append(nn.Identity())

        self.classifier = nn.Linear(hid_dim, n_classes)

    def _embed(self, x, edge_index):
        if self.n_layers == 0:
            x = F.relu(self.layers[0](x))
            return F.dropout(x, p=self.dropout, training=self.training)
        for conv, proj in zip(self.layers, self.res_projs):
            residual = proj(x)
            x = F.relu(conv(x, edge_index)) + residual
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, batch=None):
        h = self._embed(x, edge_index)
        if self.pool:
            h = global_mean_pool(h, batch)
        return self.classifier(h)

    def embed(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            return self._embed(x, edge_index)
