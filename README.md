# GRL Miniproject: Oversmoothing vs. Over-Squashing

**Research Question:** 

Do skip connections and graph rewiring complement each other in addressing oversmoothing and over-squashing , or does fixing the one worsen the other?

**2×2 experiment grid:**

| | Original graph | Rewired graph (SDRF) |
|---|---|---|
| **GCN** | Baseline | Rewiring only |
| **SkipGCN** | Skip only | Skip + Rewiring |

Measured on:
- **Graph A** (oversmoothing): Stochastic Block Model - dense communities, label = community index
- **Graph B** (over-squashing): Barbell graph - two dense clusters joined by a bridge path, label = clique membership

More details in the [experiments notebook](./experiments.ipynb).