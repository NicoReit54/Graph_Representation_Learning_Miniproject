"""Training and evaluation utilities for node- and graph-level tasks."""

import copy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ── Node classification ───────────────────────────────────────────────────────

def train_model(
    model,
    data: Data,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 300,
    patience: int = 20,
    device: str = "cpu"
    ) -> tuple:
    """Train a node-classification model (the SBM / oversmoothing experiments).

    Standard full-batch loop: forward pass over the whole graph, compute the loss
    on the train nodes only, backprop, repeat. We watch the validation accuracy
    and use early stopping - if it hasn't improved for `patience` epochs we stop
    and roll back to the best weights we saw. That avoids over-training and saves
    time once the model has converged.

    Returns (best_model, train_acc_history, val_acc_history), histories are returned
    for plotting.
    """
    model = model.to(device)
    data  = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc  = 0.0
    best_state    = copy.deepcopy(model.state_dict())
    patience_ctr  = 0
    train_hist, val_hist = [], []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        t_acc = _node_accuracy(model, data, data.train_mask, device)
        v_acc = _node_accuracy(model, data, data.val_mask,   device)
        train_hist.append(t_acc)
        val_hist.append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, train_hist, val_hist


def evaluate(model, data: Data, mask: torch.BoolTensor, device: str = "cpu") -> float:
    """Accuracy on whichever node subset the mask selects (e.g. data.test_mask)."""
    return _node_accuracy(model, data, mask, device)


def _node_accuracy(model, data, mask, device):
    """Fraction of masked nodes the model classifies correctly (no grad, eval mode)."""
    model.eval()
    with torch.no_grad():
        out  = model(data.x.to(device), data.edge_index.to(device))
        pred = out[mask].argmax(dim=1)
        return (pred == data.y.to(device)[mask]).float().mean().item()


# ── Graph classification ──────────────────────────────────────────────────────

def train_graph_model(
    model,
    train_graphs: list,
    val_graphs:   list,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 300,
    patience: int = 20,
    batch_size: int = 32,
    device: str = "cpu"
    ) -> tuple:
    """Train the source-to-target signal-propagation model (over-squashing task).

    This is the mini-batched cousin of train_model, with one important difference in
    how the loss is computed. We DON'T pool the whole graph - instead we read off
    the prediction at one specific node per graph: the target node (stored in
    data.target_node). That's on purpose: the answer is only available at the
    source node, so to get the target node right the signal genuinely has to
    travel across the bridge. A global pool would let the model peek at the
    source directly and "cheat".

    The line to consider is the target index:

        target_idx = batch.ptr[:-1] + batch.target_node.squeeze(-1)

    When PyG batches graphs it concatenates all their nodes into one big tensor.
    batch.ptr tells us where each graph starts in that tensor, so adding the
    per-graph target offset gives the global row of each graph's target node.

    Everything else (Adam, early stopping, best-weight rollback) is the same as
    train_model. 
    
    Returns (best_model, train_acc_history, val_acc_history).
    """
    model = model.to(device)
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_state   = copy.deepcopy(model.state_dict())
    patience_ctr = 0
    train_hist, val_hist = [], []

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch.x, batch.edge_index)          # (total_nodes, n_classes)
            # global index of each graph's target node = graph_start + target_node
            target_idx = batch.ptr[:-1] + batch.target_node.squeeze(-1)
            loss = F.cross_entropy(out[target_idx], batch.y)
            loss.backward()
            optimizer.step()

        t_acc = _target_accuracy(model, train_loader, device)
        v_acc = _target_accuracy(model, val_loader,   device)
        train_hist.append(t_acc)
        val_hist.append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, train_hist, val_hist


def evaluate_graph_model(model, graphs: list, batch_size: int = 32,
                         device: str = "cpu") -> float:
    """Target-node accuracy over a list of graphs (e.g. the held-out test set)."""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    return _target_accuracy(model, loader, device)


def _target_accuracy(model, loader, device):
    """Fraction of graphs whose target node gets the right label.

    Same target-node indexing trick as in train_graph_model - we only score the
    one designated readout node per graph, not all nodes.
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch      = batch.to(device)
            out        = model(batch.x, batch.edge_index)
            target_idx = batch.ptr[:-1] + batch.target_node.squeeze(-1)
            pred       = out[target_idx].argmax(dim=1)
            correct   += (pred == batch.y).sum().item()
            total     += batch.y.size(0)
    return correct / total if total > 0 else 0.0
