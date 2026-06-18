"""Metrics for measuring oversmoothing and over-squashing."""

import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def mean_average_distance(embeddings: torch.Tensor) -> float:
    """Mean Average Distance (MAD) — our main "how oversmoothed is this?" metric.

    Intuition: if oversmoothing occurs, every node ends up with basically
    the same embedding, so the "distance" between any two of them is ~0. If the
    nodes are still nicely distinguishable, those distances should stay large.

    So we just take every pair of node embeddings, measure the cosine distance
    (1 - cosine similarity), and average. Higher MAD = more distinguishable =
    less oversmoothing. A MAD near 0 is the tell-tale sign of collapse.
    """
    emb = F.normalize(embeddings, p=2, dim=1)  # (N, d)
    # cosine similarity matrix
    sim = emb @ emb.T  # (N, N)
    dist = 1.0 - sim
    # exclude diagonal (well, self-distance = 0)
    n = emb.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=emb.device)
    return dist[mask].mean().item()


def dirichlet_energy(embeddings: torch.Tensor, edge_index: torch.LongTensor) -> float:
    """Dirichlet energy — a second view on oversmoothing.

    Where MAD looks at all pairs, this one only looks at connected nodes:
    for every edge (u, v) we measure how different the two endpoints' embeddings
    are, square it, and average over edges.

        E(H) = mean over edges (u,v) of  ||h_u - h_v||^2

    If neighbours all look the same (oversmoothed), this energy collapses toward
    0. So, same story as MAD:
        low = smoothed, high = still lots of variation.
    """
    src, dst = edge_index
    diff = embeddings[src] - embeddings[dst]
    return (diff ** 2).sum(dim=1).mean().item()


def jacobian_sensitivity(
    model,
    data: Data,
    source_nodes: list,
    target_nodes: list,
    device: str = "cpu"
    ) -> np.ndarray:
    """Measure how strongly node u's input influences node v's output.

    This is one way mentioned in Topping et al. (2022) (and probably others)
    to detect over-squashing. 
    
    We take the Jacobian ||dh_v / dx_u||:
    The intutions is roughly: "If I change node u's input features a  bit, 
    how much does node v's final embedding move?". If that sensitivity goes to
    near-zero as u and v get further apart, the network physically cannot pass
    long-range signals: that's over-squashing.

    Or as the slides put it:
    The influence of the initial node features on final features decays 
    exponentially with for non-trivial computation graphs.

    How we compute it: do one forward pass, then for each (u, v) pair backprop
    a scalar summary of v's output and read off the gradient w.r.t. u's input.

    Args:
        source_nodes: list of node indices u (the input side we perturb).
        target_nodes: list of node indices v (the output side we read), same length.

    Returns:
        Array of Jacobian norms, one per (u, v) pair.
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
    device: str = "cpu"
    ) -> dict:
    """Build the over-squashing curve: average sensitivity at each hop distance.

    This is the helper that actually produces the Exp C plot. We walk the graph,
    randomly grab node pairs and bucket them by their shortest-path distance
    (1 hop, 2 hops, ...), collecting up to n_pairs pairs per bucket. Then for each
    distance we average the Jacobian sensitivity of its pairs.

    The result is "how much influence survives over d hops?" as a function of d.
    A non-squashed model decays slowly; an over-squashed one decays quickly.

    Funfact: after SDRF rewiring the graph "diameter" shrinks, so the larger
    distance buckets simply stay empty — that's why the rewired curves are short.

    Returns dict: distance -> mean sensitivity (distances with no pairs are skipped).
    """

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
