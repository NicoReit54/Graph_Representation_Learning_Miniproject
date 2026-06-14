"""SDRF-style graph rewiring via Ollivier-Ricci curvature.

Uses the GraphRicciCurvature library:
  pip install GraphRicciCurvature

Reference: Topping et al. (2022) "Understanding Over-Squashing and Bottlenecks
on Graphs via Curvature."
"""

import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx


def compute_ricci_curvature(data: Data) -> dict:
    """Compute Ollivier-Ricci curvature for every edge. Returns dict (u,v)->kappa."""
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    G = to_networkx(data, to_undirected=True)
    orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
    orc.compute_ricci_curvature()
    curvatures = {}
    for u, v, d in orc.G.edges(data=True):
        curvatures[(u, v)] = d.get("ricciCurvature", 0.0)
        curvatures[(v, u)] = d.get("ricciCurvature", 0.0)
    return curvatures


def sdrf_rewire(
    data: Data,
    n_iterations: int = 50,
    tau: float = 0.0,
    add_edges: bool = True,
    remove_edges: bool = False,
) -> Data:
    """Stochastic Discrete Ricci Flow rewiring.

    Args:
        data: input PyG Data object (undirected)
        n_iterations: number of rewiring steps
        tau: curvature threshold — edges with kappa < tau are candidates for widening
        add_edges: add shortcut edges at bottlenecks (fixes over-squashing)
        remove_edges: remove highly positive-curvature edges (reduces redundancy)

    Returns:
        New Data object with rewired edge_index. Node features/labels unchanged.
    """
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    G = to_networkx(data, to_undirected=True)
    # preserve node attributes for reconstruction
    n = data.num_nodes

    for _ in range(n_iterations):
        orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
        orc.compute_ricci_curvature()
        G_cur = orc.G

        edges_by_curvature = sorted(
            G_cur.edges(data=True),
            key=lambda e: e[2].get("ricciCurvature", 0.0)
        )

        made_change = False

        if add_edges and edges_by_curvature:
            u, v, d = edges_by_curvature[0]  # most negatively curved
            kappa = d.get("ricciCurvature", 0.0)
            if kappa < tau:
                # find pair of neighbors that maximally increases curvature
                best_pair = _best_neighbor_pair(G_cur, u, v)
                if best_pair is not None:
                    a, b = best_pair
                    if not G.has_edge(a, b):
                        G.add_edge(a, b)
                        made_change = True

        if remove_edges and edges_by_curvature:
            u, v, d = edges_by_curvature[-1]  # most positively curved
            kappa = d.get("ricciCurvature", 0.0)
            if kappa > -tau and G.degree(u) > 1 and G.degree(v) > 1:
                G.remove_edge(u, v)
                made_change = True

        if not made_change:
            break

    # rebuild PyG data with new edges
    rewired = from_networkx(G)
    new_data = Data(
        x=data.x,
        edge_index=rewired.edge_index,
        y=data.y,
        num_nodes=n,
    )
    # copy masks if present
    for attr in ["train_mask", "val_mask", "test_mask", "num_classes",
                 "target_distance", "clique_size", "bridge_length"]:
        if hasattr(data, attr):
            setattr(new_data, attr, getattr(data, attr))
    return new_data


def _best_neighbor_pair(G, u, v):
    """Return (a, b) with a in N(u), b in N(v) not already connected, or None."""
    nu = set(G.neighbors(u)) - {v}
    nv = set(G.neighbors(v)) - {u}
    candidates = [(a, b) for a in nu for b in nv if not G.has_edge(a, b) and a != b]
    if not candidates:
        return None
    # pick the pair with lowest degree sum (heuristic: prefer sparse regions)
    return min(candidates, key=lambda ab: G.degree(ab[0]) + G.degree(ab[1]))


def curvature_stats(data: Data) -> dict:
    """Return mean, min, max Ollivier-Ricci curvature over all edges."""
    curv = compute_ricci_curvature(data)
    vals = list(curv.values())
    return {
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n_negative": int(np.sum(np.array(vals) < 0)),
        "n_edges": len(vals) // 2,
    }
