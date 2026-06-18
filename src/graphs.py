"""Synthetic graph generators for controlled experiments.

Graph A (oversmoothing): Stochastic Block Model — dense intra-community edges,
  sparse inter-community edges. Label = community index.

Graph B (over-squashing): Barbell dataset — many independent barbell graphs, each
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
    """Stochastic Block Model — designed to trigger oversmoothing.

    Node features are random Gaussians (structure is what matters).
    Label = community index.
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


def make_ring_graph(
    n_nodes: int = 32,
    feature_dim: int = 8,
    seed: int = 42,
) -> Data:
    """Ring graph — used only for curvature visualization."""
    rng = np.random.default_rng(seed)
    G = nx.cycle_graph(n_nodes)
    x = torch.tensor(rng.standard_normal((n_nodes, feature_dim)) * 0.1, dtype=torch.float)
    y = torch.zeros(n_nodes, dtype=torch.long)
    data = from_networkx(G)
    data.x = x
    data.y = y
    data.num_classes = 2
    return data


def make_barbell_graph(
    clique_size: int = 5,
    bridge_length: int = 3,
    feature_dim: int = 4,
    seed: int = 42,
) -> Data:
    """Single barbell graph — used for curvature visualization only.

    Returns a node-classification Data object (label = clique membership).
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
    """Build one barbell graph for source-to-target signal propagation.

    Topology: [left clique (source)] -- bridge path -- [right clique (target)]

    Source node:  node 0  (left clique) — carries the binary signal.
    Target node:  node clique_size (first node of right clique) — must predict
                  the source label. It has no informative features.

    Distance from source to target = 1 + bridge_length hops (node 0 is 1 hop
    from the bridge entry within the left clique, then bridge_length more hops).

    Training loss and accuracy are computed ONLY at the target node, so the
    model genuinely needs to propagate the signal across the bridge.
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
    """Return a list of barbell Data objects for graph-level binary classification.

    Half the graphs have source_label=0, half have source_label=1.
    feature_dim must be >= 2 (channels 0 and 1 encode the source signal).
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
    """Split a list of graphs into train/val/test lists."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(graphs))
    n_train = int(len(perm) * train_ratio)
    n_val   = int(len(perm) * val_ratio)
    train = [graphs[i] for i in perm[:n_train]]
    val   = [graphs[i] for i in perm[n_train:n_train + n_val]]
    test  = [graphs[i] for i in perm[n_train + n_val:]]
    return train, val, test


def train_val_test_split(data: Data, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Add train/val/test masks to a node-classification Data object."""
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
