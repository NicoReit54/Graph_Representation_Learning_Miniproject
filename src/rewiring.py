"""SDRF-style graph rewiring via Ollivier-Ricci curvature.

GraphRicciCurvature segfaults when torch is also loaded (OpenMP conflict on
macOS ARM). We work around this by running the curvature computation in a
subprocess that never imports torch.

Reference: Topping et al. (2022) "Understanding Over-Squashing and Bottlenecks
on Graphs via Curvature."
"""

import json
import subprocess
import sys
import tempfile
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


# ---------------------------------------------------------------------------
# Subprocess worker — called as a script, never imports torch
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = """
import sys, json
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

payload = json.loads(sys.argv[1])
task    = payload["task"]

def _build_graph(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def _rewire_one(G, n_iter, tau, add_e, rm_e):
    for _ in range(n_iter):
        orc = OllivierRicci(G, alpha=0.0, verbose="ERROR")
        orc.compute_ricci_curvature()
        G_cur = orc.G
        sorted_edges = sorted(G_cur.edges(data=True),
                              key=lambda e: e[2].get("ricciCurvature", 0.0))
        made_change = False
        if add_e and sorted_edges:
            u, v, d = sorted_edges[0]
            if d.get("ricciCurvature", 0.0) < tau:
                nu = set(G.neighbors(u)) - {v}
                nv = set(G.neighbors(v)) - {u}
                cands = [(a, b) for a in nu for b in nv
                         if not G.has_edge(a, b) and a != b]
                if cands:
                    a, b = min(cands, key=lambda ab: G.degree(ab[0]) + G.degree(ab[1]))
                    G.add_edge(a, b)
                    made_change = True
        if rm_e and sorted_edges:
            u, v, d = sorted_edges[-1]
            if d.get("ricciCurvature", 0.0) > -tau and G.degree(u) > 1 and G.degree(v) > 1:
                G.remove_edge(u, v)
                made_change = True
        if not made_change:
            break
    return list(G.edges())

if task == "curvature":
    nodes   = payload["nodes"]
    edges   = payload["edges"]
    G = _build_graph(nodes, edges)
    orc = OllivierRicci(G, alpha=0.0, verbose="ERROR")
    orc.compute_ricci_curvature()
    result = {str(u) + "," + str(v): d.get("ricciCurvature", 0.0)
              for u, v, d in orc.G.edges(data=True)}
    print(json.dumps(result))

elif task == "rewire":
    nodes   = payload["nodes"]
    edges   = payload["edges"]
    G = _build_graph(nodes, edges)
    result = _rewire_one(G, payload["n_iter"], payload["tau"],
                         payload["add_edges"], payload["remove_edges"])
    print(json.dumps(result))

elif task == "rewire_batch":
    # Process multiple graphs in one subprocess to amortize import overhead.
    # payload["graphs"] = list of {"nodes": [...], "edges": [...]}
    results = []
    for g_spec in payload["graphs"]:
        G = _build_graph(g_spec["nodes"], g_spec["edges"])
        new_edges = _rewire_one(G, payload["n_iter"], payload["tau"],
                                payload["add_edges"], payload["remove_edges"])
        results.append(new_edges)
    print(json.dumps(results))
"""


def _call_worker(payload: dict) -> dict | list:
    """Run curvature/rewiring in a subprocess to avoid torch+OMP segfault."""
    result = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT, json.dumps(payload)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Rewiring worker failed:\n{result.stderr}")
    return json.loads(result.stdout)


def _data_to_payload(data: Data, task: str, **kwargs) -> dict:
    G = to_networkx(data, to_undirected=True)
    return {
        "task":  task,
        "nodes": list(G.nodes()),
        "edges": [list(e) for e in G.edges()],
        **kwargs,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ricci_curvature(data: Data) -> dict:
    """Return dict (u,v) -> kappa for every edge (both directions)."""
    payload = _data_to_payload(data, "curvature")
    raw = _call_worker(payload)
    curvatures = {}
    for key, val in raw.items():
        u, v = map(int, key.split(","))
        curvatures[(u, v)] = val
        curvatures[(v, u)] = val
    return curvatures


def sdrf_rewire(
    data: Data,
    n_iterations: int = 50,
    tau: float = 0.0,
    add_edges: bool = True,
    remove_edges: bool = False,
) -> Data:
    """Return a new Data object with SDRF-rewired edges. Features/labels unchanged."""
    payload = _data_to_payload(data, "rewire",
                               n_iter=n_iterations, tau=tau,
                               add_edges=add_edges, remove_edges=remove_edges)
    new_edges = _call_worker(payload)   # list of [u, v]

    # build symmetric edge_index
    src = [e[0] for e in new_edges] + [e[1] for e in new_edges]
    dst = [e[1] for e in new_edges] + [e[0] for e in new_edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    new_data = Data(x=data.x, edge_index=edge_index, y=data.y,
                    num_nodes=data.num_nodes)
    for attr in ["train_mask", "val_mask", "test_mask", "num_classes",
                 "target_distance", "clique_size", "bridge_length"]:
        if hasattr(data, attr):
            setattr(new_data, attr, getattr(data, attr))
    return new_data


def sdrf_rewire_batch(
    data_list: list,
    n_iterations: int = 20,
    tau: float = 0.0,
    add_edges: bool = True,
    remove_edges: bool = False,
) -> list:
    """Rewire multiple graphs in a single subprocess call.

    Pays the GraphRicciCurvature import overhead once instead of once per graph.
    Returns a list of new Data objects in the same order as data_list.
    """
    graphs_payload = []
    for data in data_list:
        G = to_networkx(data, to_undirected=True)
        graphs_payload.append({
            "nodes": list(G.nodes()),
            "edges": [list(e) for e in G.edges()],
        })
    payload = {
        "task": "rewire_batch",
        "graphs": graphs_payload,
        "n_iter": n_iterations,
        "tau": tau,
        "add_edges": add_edges,
        "remove_edges": remove_edges,
    }
    all_edges = _call_worker(payload)   # list of lists of [u, v]

    results = []
    for data, new_edges in zip(data_list, all_edges):
        src = [e[0] for e in new_edges] + [e[1] for e in new_edges]
        dst = [e[1] for e in new_edges] + [e[0] for e in new_edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        new_data = Data(x=data.x, edge_index=edge_index, y=data.y,
                        num_nodes=data.num_nodes)
        for attr in ["train_mask", "val_mask", "test_mask", "num_classes",
                     "target_distance", "clique_size", "bridge_length", "target_node"]:
            if hasattr(data, attr):
                setattr(new_data, attr, getattr(data, attr))
        results.append(new_data)
    return results


def curvature_stats(data: Data) -> dict:
    """Return mean/min/max Ollivier-Ricci curvature over all edges."""
    curv = compute_ricci_curvature(data)
    vals = list(curv.values())
    return {
        "mean":       float(np.mean(vals)),
        "min":        float(np.min(vals)),
        "max":        float(np.max(vals)),
        "n_negative": int(np.sum(np.array(vals) < 0)),
        "n_edges":    len(vals) // 2,
    }
