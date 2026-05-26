# LearningAVNN — Architecture Reference

**38 parameters · 1304 lines · sklearn-compatible API**

---

## Core Concept

A geometric distance classifier that learns *what kind of distance* matters for a given dataset — not just how far a point is from a class centroid, but whether its feature profile *shape* matches.

---

## Pipeline

```
Raw features X
    → MinMax normalise to [-1, 1]          (fit on training data only)
    → Six-branch combined vector (6F wide)
    → AVM: inverse-distance to K×m learnable prototypes  ─┐
    → KNN: FAISS nearest-neighbour vote                   ─┤ → λ blend → prediction
    → λ auto-tuned post-training (scalar or entropy gate)  ─┘
```

---

## Six Branches

Independent sigmoid-gated, normalised to sum to 1. Each transforms the normalised feature vector differently; the combined vector is their weighted concatenation. `bnd` and `tac` initialised at sigmoid(0.5)≈0.62 vs others at sigmoid(0)=0.50 — empirically dominant branches get a head start. Per-feature weights `fw` learned alongside branch weights.

| # | Branch | Formula | Signal |
|---|--------|---------|--------|
| 0 | linear | `x_f · √fw` | Direction + magnitude |
| 1 | circular | `√(1-x²) · √fw` | Symmetric extremeness |
| 2 | boundary | `√(‖x‖²-2x+1)` | Cross-feature coupling via ‖x‖² |
| 3 | shape | `(x-μ)/σ` per sample | Within-sample relative profile |
| 4 | quadratic | `x² · √fw` | Nonlinear extremeness |
| 5 | tanh_arccos | `tanh(arccos(x)·0.8) · √fw` | Monotone nonlinear, centre-amplified |

---

## Learnable Prototypes

Centroids are `nn.Parameter` tensors, shape `(K, m, F)`. Moved by backprop + optional EMA stabilisation.

- `n_prototypes=1` — single centroid per class (default)
- `n_prototypes>1` — mixture of prototypes; each class has m sub-centroids. Intra-class separation pushes them apart, soft EMA assignment per epoch.

**Regularisation:**

| Term | Parameter | Effect |
|------|-----------|--------|
| Anchor | `centroid_reg` | Keeps prototypes near class means |
| Inter-class push | `centroid_sep` | Pushes class prototypes apart |
| Intra-class spread | `intra_sep` | Pushes within-class prototypes apart (m>1 only) |

---

## AVM Scoring

Inverse-distance to prototypes, summed over m per class:

$$\text{score}_k(x) = \sum_j \frac{1}{d(x, \mu_{kj}) / \tau_k + \varepsilon}$$

Distance computed in the 6F combined vector space. At inference, upgraded to full **RDA Mahalanobis**:

$$\Sigma_k = (1-\alpha)\,S_k + \alpha\,S_{\text{pooled}}$$

where α=0 is per-class QDA and α=1 is pooled LDA. `mahal_alpha=0.5` balances the two. Cholesky inversion in float64 for numerical stability.

---

## Interaction Scoring

Optional additive signal blended post-AVM. Both operate in **deviation space** — sample and centroid centred by their own feature means before scoring — eliminating false penalties from absolute scale differences.

### Triangle (`use_triangle`)

Pairwise deviation cross-product for all F(F-1)/2 pairs. For pair (a, b):

$$\text{cross}_{ab,k} = (x_a - \bar{x})(\mu_{kb} - \bar{\mu}_k) - (x_b - \bar{x})(\mu_{ka} - \bar{\mu}_k)$$

Zero when the sample's relative feature profile matches the centroid's for this pair. Each pair has a learnable weight — top pairs are directly inspectable.

**Proven:** +0.058 macro F1 on moderate imbalance (4/82/14%), +0.047 on severe (60/25/10/5%).

### Bilinear (`use_bilinear`)

Learned F×F asymmetric interaction matrix W:

$$B_k(x) = \sqrt{\sum_{a,b} W_{ab}^2 \cdot (x_a - \bar{x})^2 \cdot (\mu_{kb} - \bar{\mu}_k)^2}$$

Captures directed interactions: "feature a high in sample × feature b high in centroid." Distinct from triangle (symmetric) — adds directed second-order signal. W heatmap is directly interpretable.

**Blend:** both are normalised to [0,1] and blended with geometric AVM score via `tri_weight` / `bilinear_weight`.

---

## Lambda Gate

Post-training AVM/KNN blend.

| Mode | Behaviour |
|------|-----------|
| `entropy_lambda=False` | Scalar λ grid-searched on val set from `lam_floor` to 1.0 |
| `entropy_lambda=True` | Per-sample λ(x) = lam_confident − (lam_confident − lam_uncertain) · H(x) |

Entropy gate: when AVM is confident (low entropy) → high λ, trust global structure. When uncertain (high entropy, near decision boundary) → low λ, weight local KNN.

---

## Training Losses

| Loss | Purpose | Controlled by |
|------|---------|---------------|
| NLL + label smoothing | Classification | `label_smoothing` |
| Class-weighted NLL | Imbalance correction | `weight_cap` |
| Ordinal EMD | Penalise ordinal mistakes proportionally | `ordinal`, `ordinal_weight` |
| Supervised Contrastive | Pull same-class embeddings together | `supcon`, `supcon_weight` |
| Gram matrix ortho | Decorrelate branches | `ortho_reg` |
| Centroid anchor | Keep prototypes near class means | `centroid_reg` |
| Centroid separation | Push class prototypes apart | `centroid_sep` |
| Intra separation | Push within-class prototypes apart | `intra_sep` |

---

## Inference

```python
predict_proba(X):
    Xn       = normalise(X)              # uses train min/range
    combined = _make_combined(Xn)        # numpy, 6F wide
    avm      = Mahalanobis inverse-distance to K×m prototypes
    knn      = FAISS KNN vote
    λ        = entropy_gate(avm) or scalar
    return λ·avm + (1-λ)·knn
```

FAISS uses IVF index (approximate, fast) with flat fallback for small datasets. Triangle and bilinear scores applied at AVM step during training; at inference via `_make_combined` + numpy equivalents.

---

## Empirical Track Record

| Component | Dataset | Impact |
|-----------|---------|--------|
| Full RDA Mahalanobis | Imbalanced | +0.02–0.03 macro F1 |
| Deviation triangle | Synth-moderate (4/82/14%) | +0.058 macro F1 |
| Deviation triangle | Synth-severe (60/25/10/5%) | +0.047 macro F1 |
| tanh_arccos branch | ArrivalType (oil/gas) | 0.977 learned weight |
| boundary branch | Imbalanced datasets | Consistently 2nd highest weight |
| EMA centroids | Minority classes | Stabilises weak gradient signal |
| SupCon | Moderate imbalance | Effective when ≥2 minority/batch |
| Entropy lambda | Boundary regions | Targeted, moderate impact |

---

## Open Items

| Item | Status |
|------|--------|
| Bilinear W matrix | Implemented, not yet benchmarked |
| n_prototypes > 1 | Implemented, not yet benchmarked |
| Separate KNN feature space | Designed, not implemented |

---

## Parameter Reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| `k` | 5 | KNN neighbours |
| `lr` | 1e-3 | Adam learning rate |
| `epochs` | 300 | Max training epochs |
| `batch_size` | 256 | Mini-batch size |
| `patience` | 60 | Early stopping patience |
| `val_fraction` | 0.15 | Val split for early stopping + λ search |
| `label_smoothing` | 0.05 | NLL label smoothing |
| `weight_cap` | 1.5 | Max class weight for imbalance |
| `weight_decay` | 1e-4 | Adam L2 regularisation |
| `feat_weight_reg` | 0.05 | Per-feature weight regularisation |
| `centroid_reg` | 0.1 | Anchor strength |
| `centroid_sep` | 0.05 | Inter-class push |
| `n_prototypes` | 1 | Prototypes per class |
| `intra_sep` | 0.05 | Intra-class spread (m>1) |
| `use_triangle` | False | Deviation triangle scoring |
| `tri_weight` | 0.5 | Triangle blend weight |
| `use_bilinear` | False | Bilinear W matrix scoring |
| `bilinear_weight` | 0.3 | Bilinear blend weight |
| `ortho_reg` | 0.01 | Branch decorrelation |
| `mahalanobis` | True | Full RDA at inference |
| `mahal_reg` | 1e-6 | Covariance regularisation |
| `mahal_alpha` | 0.5 | QDA↔LDA blend |
| `lam_floor` | 0.5 | Min AVM weight in λ search |
| `entropy_lambda` | False | Per-sample λ gate |
| `lam_confident` | 0.90 | λ when AVM entropy is low |
| `lam_uncertain` | 0.40 | λ when AVM entropy is high |
| `ordinal` | False | EMD ordinal loss |
| `ordinal_weight` | 0.5 | EMD loss blend |
| `per_class_tau` | True | Per-class temperature |
| `val_macro_bias` | 0.5 | 0=accuracy, 1=macro F1 for val scoring |
| `use_ivf` | True | FAISS IVF (approximate, fast) |
| `supcon` | False | Supervised contrastive loss |
| `supcon_weight` | 0.3 | SupCon loss blend |
| `supcon_temp` | 0.1 | SupCon temperature |
| `ema_centroids` | False | EMA centroid stabilisation |
| `ema_beta` | 0.9 | EMA decay rate |
| `random_state` | None | Reproducibility seed |
| `verbose` | False | Training progress output |
