"""Training and evaluation utilities."""

import copy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def train_model(
    model,
    data: Data,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 300,
    patience: int = 20,
    device: str = "cpu",
) -> tuple:
    """Train a node classification model with early stopping.

    Returns (best_model, train_acc_history, val_acc_history)
    """
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    train_hist, val_hist = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_acc = _accuracy(model, data, data.train_mask, device)
        val_acc = _accuracy(model, data, data.val_mask, device)
        train_hist.append(train_acc)
        val_hist.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, train_hist, val_hist


def evaluate(model, data: Data, mask: torch.BoolTensor, device: str = "cpu") -> float:
    return _accuracy(model, data, mask, device)


def _accuracy(model, data, mask, device):
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out[mask].argmax(dim=1)
        target = data.y.to(device)[mask]
        return (pred == target).float().mean().item()
