import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(nn.Module):
    """A plain-vanilla Graph Convolutional Network as in the practicals.

    The idea: stack n_layers of GCNConv. Each layer lets every node mix in a bit
    of its neighbours' features (with ReLU + dropout in between). After n_layers,
    every node "sees" information from up to n_layers hops away.

    This should be exactly the model that would suffer from oversmoothing: 
    stack too many layers and all the node embeddings blur into the same vector.

    Set pool=True for graph-level tasks (adds a global mean pool before the
    classifier so we get one prediction per graph instead of one per node).
    """

    def __init__(self, input_dim: int, hid_dim: int, n_classes: int,
                 n_layers: int, dropout: float = 0.3, pool: bool = False):
        """Build the network.

        Args:
            input_dim:  number of input features per node.
            hid_dim:    width of each hidden layer.
            n_classes:  number of output classes (size of the final prediction).
            n_layers:   how deep we go = how many hops a node can reach.
            dropout:    dropout probability applied after each layer.
            pool:       True for graph-level prediction, False for node-level.

        Funfact: n_layers=0 is a special case - we fall back to a single Linear
        layer (a plain MLP with no message passing at all). Handy as a "what if
        we ignore the graph entirely?" sanity check.
        """
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
        """Run the message-passing layers and return the final node embeddings.

        This is the "feature extractor" half of the model - everything except the
        last classifier layer. We split it out so we can grab the embeddings on
        their own (see embed()) to measure oversmoothing (MAD, Dirichlet energy).
        """
        if self.n_layers == 0:
            x = F.relu(self.layers[0](x))
            return F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.layers:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, batch=None):
        """Full forward pass: embed the nodes, optionally pool, then classify.

        batch is only needed when pool=True - it tells global_mean_pool which
        node belongs to which graph in a batched mini-batch.
        """
        h = self._embed(x, edge_index)
        if self.pool:
            h = global_mean_pool(h, batch)
        return self.classifier(h)

    def embed(self, x, edge_index):
        """Get the final node embeddings only (no classifier), in eval mode.

        We use this purely for analysis - feeding the embeddings into our
        oversmoothing metrics. no_grad() keeps it "cheap" since we don't train here.
        """
        self.eval()
        with torch.no_grad():
            return self._embed(x, edge_index)


class SkipGCN(nn.Module):
    """A GCN with residual (skip) connections - our fix for oversmoothing.

    Adapted from Practical 3. The only but powerful change vs. plain GCN is
    that each layer adds back its own input:

        H^(t) = ReLU(GCNConv(H^(t-1))) + proj(H^(t-1))

    That  "+ proj(H^(t-1))" is the whole trick. It gives every node a
    direct shortcut to its previous-layer features, so even very deep networks
    can keep nodes a bit more distinguishable instead of smoothing them all together
    right away.

    proj is just a shape-matcher: a Linear when the input/output widths differ
    (e.g. the first layer), and a no-op Identity when they already match.

    Set pool=True for graph-level classification.
    """

    def __init__(self, input_dim: int, hid_dim: int, n_classes: int,
                 n_layers: int, dropout: float = 0.3, pool: bool = False):
        """Same arguments as GCN - see GCN.__init__ for the full description.

        The extra piece here is res_projs: one residual projection per layer,
        built alongside the conv layers so the skip connection always lines up
        dimensionally with the conv output.
        """
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
        """Same as GCN._embed, but each layer adds the skip connection.

        For every layer we first stash the (projected) input as `residual`, run
        the usual conv + ReLU, then add the residual back on. That added term is
        what keeps deep SkipGCNs from oversmoothing.
        """
        if self.n_layers == 0:
            x = F.relu(self.layers[0](x))
            return F.dropout(x, p=self.dropout, training=self.training)
        for conv, proj in zip(self.layers, self.res_projs):
            residual = proj(x)
            x = F.relu(conv(x, edge_index)) + residual
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, batch=None):
        """Full forward pass - identical structure to GCN.forward."""
        h = self._embed(x, edge_index)
        if self.pool:
            h = global_mean_pool(h, batch)
        return self.classifier(h)

    def embed(self, x, edge_index):
        """Final node embeddings only (no classifier), for the smoothing metrics."""
        self.eval()
        with torch.no_grad():
            return self._embed(x, edge_index)
