"""Run all experiments and save figures to disk."""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))
os.chdir(_HERE)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from models import GCN, SkipGCN
from graphs import make_sbm_graph, make_ring_graph, make_barbell_graph, train_val_test_split
from rewiring import sdrf_rewire, curvature_stats, compute_ricci_curvature
from metrics import mean_average_distance, dirichlet_energy, sensitivity_vs_distance
from training import train_model, evaluate

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED    = 42
HID_DIM = 64
LAYERS  = [2, 4, 8, 16]
TRAIN_PARAMS = dict(lr=0.01, weight_decay=5e-4, epochs=300, patience=20, device=DEVICE)

COLORS = {
    'GCN': '#4878CF', 'SkipGCN': '#D65F5F',
    'GCN+Rewire': '#6ACC65', 'Skip+Rewire': '#B47CC7',
}
LINESTYLES = {'GCN': '-', 'SkipGCN': '--', 'GCN+Rewire': '-.', 'Skip+Rewire': ':'}

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f'Device: {DEVICE}')

# ── 1. Build graphs ────────────────────────────────────────────────────────────
print('\n=== Building graphs ===')

data_sbm = train_val_test_split(
    make_sbm_graph(4, 50, 0.7, 0.02, 16, SEED), seed=SEED)
print('SBM:', data_sbm)

# Barbell: primary over-squashing graph (bridge_length=5 for main experiment)
data_barbell = train_val_test_split(
    make_barbell_graph(clique_size=20, bridge_length=5, feature_dim=8, seed=SEED), seed=SEED)
print('Barbell:', data_barbell)

# Ring: curvature visualization only
data_ring = make_ring_graph(n_nodes=32, feature_dim=8, seed=SEED)
print('Ring (curvature viz only):', data_ring)

# ── 2. Rewiring ────────────────────────────────────────────────────────────────
print('\n=== Rewiring ===')

print('Rewiring SBM...')
data_sbm_rew = sdrf_rewire(data_sbm, n_iterations=30, tau=0.0)
print(f'  edges: {data_sbm.edge_index.size(1)//2} -> {data_sbm_rew.edge_index.size(1)//2}')

print('Rewiring Barbell...')
data_barbell_rew = sdrf_rewire(data_barbell, n_iterations=30, tau=0.0)
print(f'  edges: {data_barbell.edge_index.size(1)//2} -> {data_barbell_rew.edge_index.size(1)//2}')

print('Rewiring Ring (for viz)...')
data_ring_rew = sdrf_rewire(data_ring, n_iterations=20, tau=0.0)
print(f'  edges: {data_ring.edge_index.size(1)//2} -> {data_ring_rew.edge_index.size(1)//2}')

# Curvature stats
for name, g in [('SBM', data_sbm), ('Barbell', data_barbell), ('Ring', data_ring)]:
    s = curvature_stats(g)
    print(f'{name:8s}  mean κ={s["mean"]:+.3f}  min κ={s["min"]:+.3f}  '
          f'n_negative={s["n_negative"]}/{s["n_edges"]}')

# Curvature distribution plot — Ring is cleanest illustration (uniform structure)
print('\nPlotting curvature distributions...')
for graph_name, g_orig, g_rew in [
    ('Ring',    data_ring,    data_ring_rew),
    ('Barbell', data_barbell, data_barbell_rew),
]:
    curv_before = list(compute_ricci_curvature(g_orig).values())
    curv_after  = list(compute_ricci_curvature(g_rew).values())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, vals, title in zip(axes,
                                [curv_before, curv_after],
                                ['Before rewiring', 'After SDRF rewiring']):
        ax.hist(vals, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(0, color='tomato', lw=1.5, linestyle='--', label='κ = 0')
        ax.set_xlabel('Ollivier-Ricci curvature κ', fontsize=11)
        ax.set_ylabel('Edge count', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
    plt.suptitle(f'Curvature distribution — {graph_name} graph', fontsize=13, y=1.02)
    plt.tight_layout()
    fname = f'curvature_{graph_name.lower()}'
    plt.savefig(f'{fname}.pdf', bbox_inches='tight')
    plt.savefig(f'{fname}.png', bbox_inches='tight', dpi=150)
    print(f'Saved {fname}.pdf/.png')

# ── 3. Experiment A: Oversmoothing (SBM) ──────────────────────────────────────
print('\n=== Experiment A: Oversmoothing on SBM ===')

conditions_sbm = {
    'GCN':         (GCN,     data_sbm),
    'SkipGCN':     (SkipGCN, data_sbm),
    'GCN+Rewire':  (GCN,     data_sbm_rew),
    'Skip+Rewire': (SkipGCN, data_sbm_rew),
}
in_dim_sbm = data_sbm.x.size(1)
n_cls_sbm  = data_sbm.num_classes

results_mad     = {n: [] for n in conditions_sbm}
results_de      = {n: [] for n in conditions_sbm}
results_acc_sbm = {n: [] for n in conditions_sbm}

for K in LAYERS:
    print(f'\n  K={K}')
    for name, (Cls, graph) in conditions_sbm.items():
        torch.manual_seed(SEED)
        model = Cls(in_dim_sbm, HID_DIM, n_cls_sbm, K)
        model, _, _ = train_model(model, graph, **TRAIN_PARAMS)
        emb = model.embed(graph.x.to(DEVICE), graph.edge_index.to(DEVICE)).cpu()
        mad = mean_average_distance(emb)
        de  = dirichlet_energy(emb, graph.edge_index)
        acc = evaluate(model, graph, graph.test_mask, DEVICE)
        results_mad[name].append(mad)
        results_de[name].append(de)
        results_acc_sbm[name].append(acc)
        print(f'    {name:15s}  MAD={mad:.4f}  DE={de:.4f}  Acc={acc:.3f}')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (d, ylabel, title) in zip(axes, [
    (results_mad,     'Mean Average Distance (MAD) ↑', 'MAD'),
    (results_de,      'Dirichlet Energy ↑',             'Dirichlet Energy'),
    (results_acc_sbm, 'Test Accuracy ↑',                'Test Accuracy'),
]):
    for name, vals in d.items():
        ax.plot(LAYERS, vals, marker='o', label=name,
                color=COLORS[name], linestyle=LINESTYLES[name], lw=2)
    ax.set_xlabel('Layers K', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(LAYERS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
plt.suptitle('Experiment A: Oversmoothing — SBM graph', fontsize=13)
plt.tight_layout()
plt.savefig('exp_a_oversmoothing.pdf', bbox_inches='tight')
plt.savefig('exp_a_oversmoothing.png', bbox_inches='tight', dpi=150)
print('\nSaved exp_a_oversmoothing.pdf/.png')

# ── 4. Experiment B: Over-squashing (Barbell) ─────────────────────────────────
print('\n=== Experiment B: Over-Squashing on Barbell ===')
print('B.1 — Vary depth K (fixed bridge_length=5)')

conditions_barbell = {
    'GCN':         (GCN,     data_barbell),
    'SkipGCN':     (SkipGCN, data_barbell),
    'GCN+Rewire':  (GCN,     data_barbell_rew),
    'Skip+Rewire': (SkipGCN, data_barbell_rew),
}
in_dim_bb = data_barbell.x.size(1)
n_cls_bb  = data_barbell.num_classes
results_acc_bb = {n: [] for n in conditions_barbell}

for K in LAYERS:
    print(f'\n  K={K}')
    for name, (Cls, graph) in conditions_barbell.items():
        torch.manual_seed(SEED)
        model = Cls(in_dim_bb, HID_DIM, n_cls_bb, K)
        model, _, _ = train_model(model, graph, **TRAIN_PARAMS)
        acc = evaluate(model, graph, graph.test_mask, DEVICE)
        results_acc_bb[name].append(acc)
        print(f'    {name:15s}  Acc={acc:.3f}')

print('\nB.2 — Vary bridge_length (fixed K=8)')
BRIDGE_LENGTHS = [1, 2, 3, 5, 7, 10]
FIXED_K = 8
results_acc_bridge = {n: [] for n in ['GCN', 'SkipGCN', 'GCN+Rewire', 'Skip+Rewire']}

for bl in BRIDGE_LENGTHS:
    print(f'\n  bridge_length={bl}')
    g = train_val_test_split(
        make_barbell_graph(clique_size=20, bridge_length=bl, feature_dim=8, seed=SEED),
        seed=SEED)
    g_rew = sdrf_rewire(g, n_iterations=30, tau=0.0)
    for name, (Cls, graph) in [
        ('GCN',         (GCN,     g)),
        ('SkipGCN',     (SkipGCN, g)),
        ('GCN+Rewire',  (GCN,     g_rew)),
        ('Skip+Rewire', (SkipGCN, g_rew)),
    ]:
        torch.manual_seed(SEED)
        model = Cls(g.x.size(1), HID_DIM, g.num_classes, FIXED_K)
        model, _, _ = train_model(model, graph, **TRAIN_PARAMS)
        acc = evaluate(model, graph, graph.test_mask, DEVICE)
        results_acc_bridge[name].append(acc)
        print(f'    {name:15s}  Acc={acc:.3f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
for name, vals in results_acc_bb.items():
    ax.plot(LAYERS, vals, marker='o', label=name,
            color=COLORS[name], linestyle=LINESTYLES[name], lw=2)
ax.set_xlabel('Layers K', fontsize=11)
ax.set_ylabel('Test Accuracy ↑', fontsize=11)
ax.set_title('Accuracy vs Depth  (bridge_length=5)', fontsize=12)
ax.set_xticks(LAYERS); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
for name, vals in results_acc_bridge.items():
    ax.plot(BRIDGE_LENGTHS, vals, marker='o', label=name,
            color=COLORS[name], linestyle=LINESTYLES[name], lw=2)
ax.axvline(FIXED_K - 2, color='gray', lw=1, linestyle=':', alpha=0.7,
           label=f'K−2={FIXED_K-2} (reach limit)')
ax.set_xlabel('Bridge length (hops between cliques)', fontsize=11)
ax.set_ylabel('Test Accuracy ↑', fontsize=11)
ax.set_title(f'Accuracy vs Bridge Length  (K={FIXED_K})', fontsize=12)
ax.set_xticks(BRIDGE_LENGTHS); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle('Experiment B: Over-Squashing — Barbell graph', fontsize=13)
plt.tight_layout()
plt.savefig('exp_b_oversquashing.pdf', bbox_inches='tight')
plt.savefig('exp_b_oversquashing.png', bbox_inches='tight', dpi=150)
print('\nSaved exp_b_oversquashing.pdf/.png')

# ── 5. Experiment C: Jacobian sensitivity (Barbell) ───────────────────────────
print('\n=== Experiment C: Jacobian Sensitivity vs Distance (Barbell) ===')
JACOBIAN_K = 8
jacobian_results = {}

for name, (Cls, graph) in [
    ('GCN',         (GCN,     data_barbell)),
    ('SkipGCN',     (SkipGCN, data_barbell)),
    ('GCN+Rewire',  (GCN,     data_barbell_rew)),
    ('Skip+Rewire', (SkipGCN, data_barbell_rew)),
]:
    torch.manual_seed(SEED)
    model = Cls(data_barbell.x.size(1), HID_DIM, data_barbell.num_classes, JACOBIAN_K)
    model, _, _ = train_model(model, graph, **TRAIN_PARAMS)
    sens = sensitivity_vs_distance(model, graph, n_pairs=30, max_distance=15, device=DEVICE)
    jacobian_results[name] = sens
    print(f'  {name}: {sens}')

fig, ax = plt.subplots(figsize=(8, 4))
for name, sens_dict in jacobian_results.items():
    dists = sorted(sens_dict.keys())
    vals  = [sens_dict[d] for d in dists]
    ax.plot(dists, vals, marker='o', label=name,
            color=COLORS[name], linestyle=LINESTYLES[name], lw=2)
ax.set_xlabel('Graph distance between node pair', fontsize=11)
ax.set_ylabel('Mean Jacobian norm  ||∂h_v / ∂x_u||', fontsize=11)
ax.set_title(f'Jacobian sensitivity vs distance  (K={JACOBIAN_K}, Barbell)', fontsize=12)
ax.set_yscale('log')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('exp_c_jacobian.pdf', bbox_inches='tight')
plt.savefig('exp_c_jacobian.png', bbox_inches='tight', dpi=150)
print('Saved exp_c_jacobian.pdf/.png')

# ── 6. Summary table ──────────────────────────────────────────────────────────
print('\n=== Summary (K=8) ===')
K_IDX = LAYERS.index(8)
BL_IDX = BRIDGE_LENGTHS.index(5)
header = f"{'Model':15s}  {'MAD(SBM)':>10s}  {'Acc(SBM)':>10s}  {'Acc(Barbell bl=5)':>18s}"
print(header)
print('-' * len(header))
for name in ['GCN', 'SkipGCN', 'GCN+Rewire', 'Skip+Rewire']:
    print(f"{name:15s}  {results_mad[name][K_IDX]:>10.4f}  "
          f"{results_acc_sbm[name][K_IDX]:>10.3f}  "
          f"{results_acc_bridge[name][BL_IDX]:>18.3f}")

print('\nDone! All figures saved.')
