"""Metrics for measuring oversmoothing and over-squashing."""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data


def mean_average_distance(embeddings: torch.Tensor) -> float:
    """Mean Average Distance (MAD) between all pairs of node embeddings.

    Higher = more distinguishable = less oversmoothing.
    Computed as mean pairwise cosine distance (1 - cosine_similarity).
    """
    emb = F.normalize(embeddings, p=2, dim=1)  # (N, d)
    # cosine similarity matrix
    sim = emb @ emb.T  # (N, N)
    dist = 1.0 - sim
    # exclude diagonal (self-distance = 0)
    n = emb.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=emb.device)
    return dist[mask].mean().item()


def dirichlet_energy(embeddings: torch.Tensor, edge_index: torch.LongTensor) -> float:
    """Dirichlet energy: sum of squared differences across edges.

    E(H) = sum_{(u,v) in E} ||h_u - h_v||^2

    A complementary smoothing measure — low energy = oversmoothed.
    """
    src, dst = edge_index
    diff = embeddings[src] - embeddings[dst]
    return (diff ** 2).sum(dim=1).mean().item()


def jacobian_sensitivity(
    model,
    data: Data,
    source_nodes: list,
    target_nodes: list,
    device: str = "cpu",
) -> np.ndarray:
    """Compute ||dh_v^(K) / dx_u||_F for pairs (u, v).

    Args:
        source_nodes: list of node indices u (input side)
        target_nodes: list of node indices v (output side), same length

    Returns:
        Array of Jacobian norms, shape (len(source_nodes),)
    """
    model.train()  # need gradients
    x = data.x.to(device).requires_grad_(True)
    edge_index = data.edge_index.to(device)

    # forward pass to get all embeddings
    logits = model(x, edge_index)  # (N, C)

    norms = []
    for u, v in zip(source_nodes, target_nodes):
        # scalar output: sum of logits at v
        scalar = logits[v].sum()
        grad = torch.autograd.grad(scalar, x, retain_graph=True)[0]  # (N, d_in)
        norm = grad[u].norm().item()
        norms.append(norm)

    model.eval()
    return np.array(norms)


def sensitivity_vs_distance(
    model,
    data: Data,
    n_pairs: int = 50,
    max_distance: int = 10,
    device: str = "cpu",
) -> dict:
    """Sample node pairs at each graph distance and compute mean Jacobian norm.

    Returns dict: distance -> mean sensitivity
    """
    import networkx as nx
    from torch_geometric.utils import to_networkx

    G = to_networkx(data, to_undirected=True)
    nodes = list(G.nodes())
    rng = np.random.default_rng(0)

    distance_to_pairs = {d: [] for d in range(1, max_distance + 1)}
    attempts = 0
    while any(len(v) < n_pairs for v in distance_to_pairs.values()) and attempts < 5000:
        u, v = rng.choice(nodes, 2, replace=False)
        try:
            d = nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            attempts += 1
            continue
        if d in distance_to_pairs and len(distance_to_pairs[d]) < n_pairs:
            distance_to_pairs[d].append((u, v))
        attempts += 1

    results = {}
    for d, pairs in distance_to_pairs.items():
        if not pairs:
            continue
        srcs = [p[0] for p in pairs]
        tgts = [p[1] for p in pairs]
        norms = jacobian_sensitivity(model, data, srcs, tgts, device)
        results[d] = float(np.mean(norms))

    return results
