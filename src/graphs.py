"""Synthetic graph generators for controlled experiments.

Graph A (oversmoothing): Stochastic Block Model — dense intra-community edges,
  sparse inter-community edges. Label = community index.
  Message passing converges fast because neighborhoods are highly similar.

Graph B (over-squashing): Ring / Barbell graphs where the label of node v
  depends on a node exactly d hops away. Vary d to control difficulty.
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

    # random node features
    x = torch.tensor(rng.standard_normal((n, feature_dim)), dtype=torch.float)
    y = torch.tensor([G.nodes[v]["block"] for v in G.nodes], dtype=torch.long)

    data = from_networkx(G)
    data.x = x
    data.y = y
    data.num_classes = n_communities
    return data


def make_ring_graph(
    n_nodes: int = 64,
    feature_dim: int = 8,
    target_distance: int = 8,
    seed: int = 42,
) -> Data:
    """Ring graph — designed to trigger over-squashing.

    Node v has label = feature of the node exactly `target_distance` hops away.
    The minimum path from any node to its target is exactly target_distance,
    forcing information to travel through a bottleneck.

    target_distance controls difficulty: larger d = worse squashing.
    """
    rng = np.random.default_rng(seed)
    assert target_distance < n_nodes // 2, "target_distance must be < n_nodes/2"

    G = nx.cycle_graph(n_nodes)

    # binary features: node i has feature class = i % 2 (simple distinguishing signal)
    # actual label of node v = feature class of node (v + target_distance) % n_nodes
    raw_labels = torch.tensor([i % 2 for i in range(n_nodes)], dtype=torch.long)
    y = torch.tensor(
        [(i + target_distance) % 2 for i in range(n_nodes)], dtype=torch.long
    )

    # one-hot features from raw_labels
    x = torch.zeros(n_nodes, feature_dim)
    for i in range(n_nodes):
        x[i, raw_labels[i].item()] = 1.0
    # add small noise
    x += torch.tensor(rng.standard_normal((n_nodes, feature_dim)) * 0.1,
                      dtype=torch.float)

    data = from_networkx(G)
    data.x = x
    data.y = y
    data.num_classes = 2
    data.target_distance = target_distance
    return data


def make_barbell_graph(
    clique_size: int = 20,
    bridge_length: int = 5,
    feature_dim: int = 8,
    seed: int = 42,
) -> Data:
    """Barbell graph — two cliques connected by a path.

    Label: which clique a node belongs to (0 or 1), path nodes = -1 (ignored).
    The bridge path is the topological bottleneck — high negative curvature.
    """
    rng = np.random.default_rng(seed)
    left = nx.complete_graph(clique_size)
    right = nx.complete_graph(clique_size)

    # relabel right clique nodes
    right = nx.relabel_nodes(right, {i: i + clique_size for i in right.nodes})

    G = nx.union(left, right)

    # add bridge path between clique_size-1 and clique_size
    bridge_nodes = list(range(2 * clique_size, 2 * clique_size + bridge_length - 1))
    chain = [clique_size - 1] + bridge_nodes + [clique_size]
    for a, b in zip(chain, chain[1:]):
        G.add_edge(a, b)

    n = G.number_of_nodes()
    x = torch.tensor(rng.standard_normal((n, feature_dim)) * 0.1, dtype=torch.float)

    # distinct features for left vs right clique
    for i in range(clique_size):
        x[i, 0] = 1.0
    for i in range(clique_size, 2 * clique_size):
        x[i, 1] = 1.0

    # label: 0 = left clique, 1 = right clique, -1 = bridge
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


def train_val_test_split(data: Data, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Add train/val/test masks to a Data object."""
    rng = np.random.default_rng(seed)
    n = data.num_nodes

    # exclude bridge nodes (y == -1) from split
    valid = (data.y >= 0).nonzero(as_tuple=True)[0].numpy()
    perm = rng.permutation(valid)

    n_train = int(len(perm) * train_ratio)
    n_val = int(len(perm) * val_ratio)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data
