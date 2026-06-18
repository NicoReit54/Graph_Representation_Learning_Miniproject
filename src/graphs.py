"""Synthetic graph generators for controlled experiments.

Graph A (oversmoothing): Stochastic Block Model - dense intra-community edges,
  sparse inter-community edges. Label = community index.

Graph B (over-squashing): Barbell dataset - many independent barbell graphs, each
  with a random binary source signal planted at one node of the left clique.
  The GCN must propagate that signal across the bridge via global pooling.
  Vary bridge_length to control bottleneck severity.
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data


def make_sbm_graph(
    n_communities: int = 4,
    nodes_per_community: int = 50,
    p_intra: float = 0.7,
    p_inter: float = 0.02,
    feature_dim: int = 16,
    seed: int = 42,
) -> Data:
    """Stochastic Block Model - our "make oversmoothing" graph.

    We build a few communities of nodes. Inside a community nodes are densely
    connected (p_intra high); between communities they're barely connected
    (p_inter low). The task is to predict which community each node belongs to.

    Why this should trigger oversmoothing: with so many within-community edges, 
    a deep GCN keeps averaging each node with its dense neighbourhood until every 
    node in a community - eventually the whole graph - collapses to the same vector.

    The node features themselves are just random Gaussians: we want the model to
    rely on the structure, not on easy features. Label = community index.
    """
    rng = np.random.default_rng(seed)
    n = n_communities * nodes_per_community
    sizes = [nodes_per_community] * n_communities
    probs = [[p_intra if i == j else p_inter for j in range(n_communities)]
             for i in range(n_communities)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)

    x = torch.tensor(rng.standard_normal((n, feature_dim)), dtype=torch.float)
    y = torch.tensor([G.nodes[v]["block"] for v in G.nodes], dtype=torch.long)

    data = from_networkx(G)
    data.x = x
    data.y = y
    data.num_classes = n_communities
    return data

def make_barbell_graph(
    clique_size: int = 5,
    bridge_length: int = 3,
    feature_dim: int = 4,
    seed: int = 42,
) -> Data:
    """Build one barbell graph - two cliques joined by a thin bridge.

    Shape (well who would have guessed):  
        [left clique] --- bridge path --- [right clique]

    The two dense cliques are connected only through a single narrow path, so
    every bit of information passing from one side to the other has to go
    through that bridge. That's the "textbook" bottleneck that causes
    over-squashing, which should be exactly what we want to see.

    Note:
    This version is for curvature visualization ONLY (we colour edges by Ollivier-Ricci
    curvature and watch SDRF fix the bridge). It's a node-classification Data
    object where the label = which clique a node is in. For the actual training
    task we use make_barbell_dataset() instead, which plants a signal to predict.
    """
    rng = np.random.default_rng(seed)
    left  = nx.complete_graph(clique_size)
    right = nx.complete_graph(clique_size)
    right = nx.relabel_nodes(right, {i: i + clique_size for i in right.nodes})
    G = nx.union(left, right)
    bridge_nodes = list(range(2 * clique_size, 2 * clique_size + bridge_length - 1))
    chain = [clique_size - 1] + bridge_nodes + [clique_size]
    for a, b in zip(chain, chain[1:]):
        G.add_edge(a, b)
    n = G.number_of_nodes()
    x = torch.tensor(rng.standard_normal((n, feature_dim)) * 0.1, dtype=torch.float)
    y = torch.full((n,), -1, dtype=torch.long)
    for i in range(clique_size):
        y[i] = 0
    for i in range(clique_size, 2 * clique_size):
        y[i] = 1
    data = from_networkx(G)
    data.x = x
    data.y = y
    data.num_classes = 2
    data.clique_size = clique_size
    data.bridge_length = bridge_length
    return data


def _make_barbell_instance(
    clique_size: int,
    bridge_length: int,
    feature_dim: int,
    source_label: int,
    rng: np.random.Generator,
) -> Data:
    """Build one barbell graph for the source-to-target signal-propagation task.

    This is the workhorse that turns the barbell into an actual learning problem.

    Topology: [left clique (source)] -- bridge path -- [right clique (target)]

    The setup is deliberately a "telephone across a bottleneck" game:
      - Source = node 0 (left clique). We plant a clean binary signal in its
        features (a one-hot in channel 0 or 1, depending on source_label).
      - Every other node - including the target - gets only tiny random noise,
        so there's no way to shortcut the answer from local features.
      - Target = node clique_size (first node of the right clique). This is the
        only node we ever read the prediction from.

    Distance source -> target = 1 + bridge_length hops (node 0 is 1 hop from the
    bridge entry inside the left clique, then bridge_length hops across the bridge).

    Because the loss/accuracy are scored ONLY at the target node, the model can't
    win unless the signal physically makes it all the way across the bridge - so
    longer bridges = harder over-squashing, which is the knob we sweep in Exp B.
    """
    left  = nx.complete_graph(clique_size)
    right = nx.complete_graph(clique_size)
    right = nx.relabel_nodes(right, {i: i + clique_size for i in right.nodes})
    G = nx.union(left, right)

    bridge_nodes = list(range(2 * clique_size, 2 * clique_size + bridge_length - 1))
    chain = [clique_size - 1] + bridge_nodes + [clique_size]
    for a, b in zip(chain, chain[1:]):
        G.add_edge(a, b)

    n = G.number_of_nodes()

    # all nodes: small noise features
    x = torch.tensor(rng.standard_normal((n, feature_dim)) * 0.01, dtype=torch.float)
    # source node 0: one-hot binary signal
    x[0, :] = 0.0
    x[0, source_label] = 1.0

    edge_index = from_networkx(G).edge_index

    # target_node is always the first node of the right clique
    target_node = clique_size
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([source_label], dtype=torch.long),
        num_nodes=n,
        target_node=torch.tensor([target_node], dtype=torch.long),
    )


def make_barbell_dataset(
    n_graphs: int = 400,
    clique_size: int = 5,
    bridge_length: int = 3,
    feature_dim: int = 4,
    seed: int = 42,
) -> list:
    """Make a whole dataset of barbell graphs for the over-squashing experiments.

    Just calls _make_barbell_instance() n_graphs times. We keep the classes
    balanced (half source_label=0, half =1) and then shuffle so the two classes
    aren't in a predictable order.

    Note: every graph here has the SAME topology (same clique_size / bridge_length)
    and only differs in which signal is planted + the noise. That's on purpose - it
    lets us rewire the shared topology once and reuse it for all instances (otherwise 
    we'd have to generate a new graph for every instance, which is a bit timeconsuming).

    feature_dim must be >= 2, since channels 0 and 1 are reserved for the one-hot
    source signal.
    """
    assert feature_dim >= 2
    rng = np.random.default_rng(seed)
    graphs = []
    for i in range(n_graphs):
        label = i % 2  # balanced classes
        graphs.append(_make_barbell_instance(clique_size, bridge_length, feature_dim, label, rng))
    rng.shuffle(graphs)
    return graphs


def split_dataset(graphs: list, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Split a list of graphs into train / val / test lists (for graph-level tasks).

    Used by the barbell experiments where each item is a whole graph. We shuffle
    the indices once (seeded) and slice into the three sets.
    Default: 60 / 20 / 20 split.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(graphs))
    n_train = int(len(perm) * train_ratio)
    n_val   = int(len(perm) * val_ratio)
    train = [graphs[i] for i in perm[:n_train]]
    val   = [graphs[i] for i in perm[n_train:n_train + n_val]]
    test  = [graphs[i] for i in perm[n_train + n_val:]]
    return train, val, test


def train_val_test_split(data: Data, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Add train / val / test boolean masks to a single node-classification graph.

    This is the node-level counterpart of split_dataset: instead of splitting a
    list of graphs, we split the *nodes* of one graph. The masks 
    tell the training loop which nodes to learn from, tune on, or report on.

    We only ever assign nodes with a valid label (y >= 0), so any placeholder
    "-1" nodes (e.g. the barbell bridge nodes) are quietly left out of all splits.
    """
    rng = np.random.default_rng(seed)
    n = data.num_nodes
    valid = (data.y >= 0).nonzero(as_tuple=True)[0].numpy()
    perm = rng.permutation(valid)
    n_train = int(len(perm) * train_ratio)
    n_val   = int(len(perm) * val_ratio)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask
    return data
