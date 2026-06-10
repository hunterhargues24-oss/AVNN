# LearningAVNN

**A geometric tabular classifier with learnable distance metrics, separate global/local feature spaces, and imbalance handling.**

`sklearn`-compatible · pure PyTorch training · FAISS inference · 36 parameters · ~1300 lines

---

## Table of Contents

1. [What It Is](#what-it-is)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Evolution History](#evolution-history)
5. [Benchmarks](#benchmarks)
6. [Parameter Reference](#parameter-reference)
7. [Design Decisions and What Didn't Work](#design-decisions-and-what-didnt-work)
8. [Open Items](#open-items)

---

## What It Is

LearningAVNN (Learning Angular Vector Neural Network) is a geometric distance-based tabular classifier. Instead of learning a standard decision boundary, it learns *what kind of distance* matters — transforming features into geometric spaces, then measuring how close a sample is to learned class prototypes.

The core idea is a two-space split:

- A **learned AVM space** (global structure) scores each sample by its Mahalanobis proximity to `K×m` learnable class prototypes.
- A **separate, frozen KNN space** (local structure) scores each sample by an inverse-distance vote over its nearest training neighbours via FAISS.

The two scores are late-fused: `λ·AVM + (1-λ)·KNN`, with `λ` either tuned on the validation set or gated per-sample by confidence. Because the two spaces use different branch transforms, they genuinely disagree — the fusion can correct one with the other rather than averaging two views of the same vector.

It was designed for imbalanced multi-class tabular classification, where tree ensembles and kernel methods struggle to recover rare-class signal without sacrificing majority-class accuracy.

---

## Quick Start

```python
from LearningAVNN import LearningAVNN

# Balanced dataset
model = LearningAVNN(k=5, epochs=300)

# Imbalanced dataset — moderate (2–10% minority)
model = LearningAVNN(
    ordinal=True, ordinal_weight=0.3,
    supcon=True, supcon_weight=0.3,
    ema_centroids=True,
    val_macro_bias=0.8,
    feat_weight_reg=0.01,
)

# Imbalanced dataset — severe (<2% minority)
model = LearningAVNN(
    ema_centroids=True,
    supcon=True, supcon_weight=0.3,
    mahal_alpha=0.3,
    val_macro_bias=0.9,
    k=10,
    use_ivf=True,
)

# ArrivalType — reference config (oil & gas, ~2.17M rows, ~5% rarest class)
model = LearningAVNN(
    n_prototypes=2,            # ArrivalType classes are multimodal
    ema_centroids=True,        # stabilise minority prototypes under sparse gradient
    mahal_alpha=0.3,           # lean toward per-class QDA for severe imbalance
    weight_cap=4.0,            # allow stronger minority up-weighting at this scale
    val_macro_bias=0.85,       # tune toward macro F1, not accuracy
    k=10,
    use_ivf=True,              # IVF index for the 2.17M-row training set
)

model.fit(X_train, y_train)
pred  = model.predict(X_test)
proba = model.predict_proba(X_test)
print(model.summary(feature_names))

# Optional: per-sample disagreement between the AVM and KNN views
pred, proba, disagreement = model.predict_with_uncertainty(X_test)
```

> The ArrivalType block is a working starting point, not a tuned optimum — `weight_cap`, `mahal_alpha`, `n_prototypes`, and `val_macro_bias` are the first knobs to sweep per acquisition/basin if the class mix shifts. `boundary_init=True` and `use_dual_boundary=True` are experimental geometry options worth A/B testing on this dataset (see Architecture).

---

## Architecture

### Pipeline

```
Raw X  →  MinMax normalise [-1,1]
       →  AVM space:  3-branch combined vector  (tac + boundary[·dual] + circular)
              →  Mahalanobis inverse-distance to K×m learnable prototypes
       →  KNN space:  separate 3-branch combined vector  (linear + shape + quadratic)
              →  FAISS inverse-distance vote over k nearest training points
       →  λ·AVM + (1-λ)·KNN   (λ: scalar / confidence-gate auto-selected, or entropy gate)
```

The AVM and KNN spaces are **distinct**. Only the AVM space participates in the training forward pass and receives gradient; the KNN branch weights are parameters but get no gradient, so they remain a fixed geometric view that complements the learned space. Per-feature weights `fw` (softmax-normalised to sum to `n_feat`, so uniform = 1.0 each) are learned once and shared across both spaces.

### AVM Branches — global structure (learned)

Three branches, combined into a `3F`-wide vector (`4F` if `use_dual_boundary`). Branch weights are independent normalised-sigmoid gates (not softmax), so multiple branches can be strong at once. `tanh_arccos` and `boundary` are initialised with a `+0.5` raw-logit head start over `circular`.

| Branch | Formula | Signal |
|--------|---------|--------|
| tanh_arccos | `tanh(arccos(x)·0.8)·√fw` | Monotone nonlinear; high at x≈−1, ~0 at x≈+1; most sensitive near x=0 |
| boundary | `√(‖x‖²−2x+1)` | Cross-feature coupling via the global term ‖x‖² |
| circular | `√(1−x²)·√fw` | Symmetric extremeness; peaks at ±1 |

**`use_dual_boundary`** (opt-in) extends the boundary branch to `2F` by adding the negative-face distances `√(‖x‖²+2x+1)`, interleaved with the positive-face distances. This gives the AVM a direct signal about proximity to the negative hypercube face, complementary to `tanh_arccos`.

### KNN Branches — local structure (frozen)

Three branches, combined into a separate `3F`-wide vector that feeds the FAISS index. These weights receive no gradient and stay at their uniform initialisation.

| Branch | Formula | Signal |
|--------|---------|--------|
| linear | `x·√fw` | Signed magnitude; direct local proximity |
| shape | `(x−μ)/σ` per sample | Within-sample relative profile fingerprint |
| quadratic | `x²·√fw` | Nonlinear extremeness |

### Learnable Prototypes

Prototypes are an `nn.Parameter` of shape `(K, m, F)` — `K` classes, `m` prototypes each — updated by backprop plus optional EMA stabilisation. `centroid_reg` anchors them near their initial class means; `centroid_sep` pushes prototypes of different classes apart; `intra_sep` pushes the prototypes *within* a class apart (active only when `m>1`). Separation losses are cached once per epoch and detached.

Two initialisation modes:

- **Jitter init (default)** — every prototype starts at its class mean; for `m>1`, small Gaussian jitter breaks symmetry so the NLL gradient and `intra_sep` can spread them.
- **Boundary-medoid init (`boundary_init=True`, only when `m>1`)** — prototype 0 sits at the class centroid (interior anchor, captures the bulk); prototypes 1…m−1 are placed at the lowest-positive-margin training points (frontier defenders), where margin = `d(nearest opposing centroid) − d(own centroid)`. Negative-margin points are excluded as likely label noise; degenerate classes fall back to jitter.

### AVM Scoring

For each prototype, score is an inverse distance; per-class score sums over that class's `m` prototypes, then normalises:

$$\text{score}_k(x) = \sum_{j=1}^{m} \frac{1}{d(x, \mu_{kj}) + \varepsilon}$$

In the training forward pass, `d` is Euclidean in the AVM space and a learned per-class temperature `τ` (when `per_class_tau=True`) lets minority classes adopt softer scoring.

At inference, `d` is a **full RDA Mahalanobis** distance with a per-prototype covariance:

$$\Sigma_{kj} = (1-\alpha)\,S_{kj} + \alpha\,S_{\text{pooled}}, \qquad \Sigma_{kj}^{\text{reg}} = (1-r)\,\Sigma_{kj} + r\cdot\overline{\text{diag}}\cdot I$$

`α` (`mahal_alpha`) blends per-class QDA (`α→0`) toward pooled LDA (`α→1`); `r` (`mahal_reg`) is the ridge term that keeps each matrix positive-definite when `n_k < D`. Covariances are inverted via Cholesky (with a pseudo-inverse fallback) and held in float64, since float32 overflows on high-dimensional combined vectors. When `m>1`, each prototype gets its own covariance estimated from the training points soft-assigned to it — earlier versions shared one class-pooled covariance across all of a class's prototypes, which contradicted the point of spreading them with `intra_sep`.

### Lambda Fusion

The fused posterior is `λ·AVM + (1-λ)·KNN`. There are three λ regimes:

| Mode | Behaviour |
|------|-----------|
| Scalar (auto) | Grid-searched on `linspace(lam_floor, 1.0)`; `lam=1.0` (pure AVM) is always a candidate |
| Confidence gate (auto) | Per-sample `λ = c_AVM / (c_AVM + c_KNN)`, where `c = 1 − normalised entropy`, clipped to `[lam_uncertain, lam_confident]`. High AVM confidence → lean AVM; high KNN confidence → lean KNN |
| Entropy gate (`entropy_lambda=True`) | Per-sample `λ(x) = lam_confident − (lam_confident − lam_uncertain)·H(x)`, where `H` is the normalised AVM entropy |

When `entropy_lambda=False`, `fit` evaluates both the scalar search and the confidence gate on the validation set and keeps whichever scores higher. If FAISS is unavailable the model falls back to pure AVM (`λ=1.0`).

### Training Losses

| Loss | Purpose | Key parameter |
|------|---------|---------------|
| Class-weighted NLL | Imbalance correction (balanced weights, capped) | `weight_cap` |
| Label smoothing | Prevent overconfidence (corrected to true `(1-α)·NLL + α·mean_k(-log p_k)`) | `label_smoothing` |
| Ordinal EMD | Penalise ordinal mistakes proportionally | `ordinal_weight` |
| Supervised Contrastive | Pull same-class AVM embeddings together per batch | `supcon_weight`, `supcon_temp` |
| Gram-matrix ortho | Decorrelate the AVM branches | `ortho_reg` |
| Centroid anchor | Keep prototypes near their class means | `centroid_reg` |
| Centroid separation | Push different-class prototypes apart | `centroid_sep` |
| Intra separation | Push within-class prototypes apart (`m>1`) | `intra_sep` |
| Feature-weight reg | Pull learned feature weights toward uniform | `feat_weight_reg` |

Optimisation: Adam with `CosineAnnealingWarmRestarts` (`T_0=30, T_mult=2`), gradient-norm clipping at 1.0. Early stopping validates every 5 epochs after a 20-epoch warmup, on a composite metric (`_val_score`) computed from a single confusion matrix: `(1−b)·½(acc+weighted_F1) + b·macro_F1`, where `b = val_macro_bias`.

### Inference

```python
predict_proba(X):
    Xn      = normalise(X)                          # train statistics
    avm     = Mahalanobis inverse-distance          # AVM 3-branch space, to K×m prototypes
    if no FAISS index or λ≥1.0: return avm
    knn     = FAISS KNN vote                         # separate KNN 3-branch space
    λ       = entropy gate | confidence gate | scalar
    return  λ·avm + (1-λ)·knn
```

`predict_with_uncertainty(X)` additionally returns a per-sample disagreement score (both views confident but predicting different classes) — useful for triage and active-labelling.

---

## Evolution History

### Phase 1 — Foundation

Started as a three-branch static classifier (`FastTriBranchAVNN`) with fixed branch weights: angular (arccos), euclidean (tanh_arccos), and shape (z-score). No gradient, no training — just class-mean centroids and FAISS KNN. Competitive with SVM on balanced data; weak on imbalance. FastTriBranchAVNN beat the early learnable version on Wine cultivar (0.9884 vs 0.9831) because the shape (z-score) branch captured within-sample relative profiles the learnable arccos branches couldn't — which is why a shape branch survives in the model.

### Phase 2 — Learnable Branches

Introduced gradient-based branch weights, per-feature weights, and learnable centroids. Replaced softmax branch competition with **independent normalised sigmoid gates**, eliminating the zero-sum dynamic where raising one branch necessarily suppressed others. Ablation showed dropping low-weight branches degraded holdout performance — even near-uniform branches enriched the geometry.

### Phase 3 — Distance Upgrade

Replaced diagonal covariance with **full RDA Mahalanobis** (α-blend of per-class QDA and pooled LDA), Cholesky inversion, float64 intermediates (overflow first hit on the 384-dim digits combined space).

### Phase 4 — Imbalance Toolkit

Added Supervised Contrastive loss, EMA centroid stabilisation, Ordinal EMD loss, and per-class temperature.

### Phase 5 — Mixture of Prototypes

Extended centroids from `(K, F)` to `(K, m, F)`. Soft EMA assignment updates each prototype toward its nearest sub-cluster (subsampling classes >5000 points to avoid an `O(N·m)` cdist). Targets multimodal classes such as medium-quality wine spanning q=5 and q=6.

### Phase 6 — Triangle Interaction Scoring *(later retired)*

A pairwise deviation cross-product over feature pairs, blended into the AVM score, with per-pair learnable weights. It produced real gains in its era (+0.058 macro F1 on synth-moderate, +0.047 on synth-severe) and the deviation-space formulation fixed false penalties from absolute-scale differences. It has since been **removed from the codebase** — see Phase 9.

### Phase 7 — Lambda Gate

Added the entropy-based per-sample λ gate and, alongside it, the confidence gate; `fit` auto-selects between the scalar search and the confidence gate on the validation set.

### Phase 8 — Cleanup

Removed the kernel-trick triangle variant, triplet scoring, bilinear-W scoring, branch pruning, and the MLP λ gate. Consolidated three validation weights into a single `val_macro_bias`. Corrected several latent bugs: label-smoothing math, λ grid `linspace` (so `1.0` is reachable), and early-stopping patience (skipped epochs no longer double-count).

### Phase 9 — Two-Space Split & Simplification *(current build)*

The defining change of the current version:

- **Dedicated KNN feature space.** AVM and KNN no longer share one combined vector. AVM gets `tanh_arccos + boundary + circular` (learned); KNN gets `linear + shape + quadratic` (frozen, no gradient). Previously both views were the same vector, so they couldn't genuinely disagree — now the fusion has real signal to work with.
- **Triangle scoring removed.** The pairwise interaction branch is gone; the dedicated KNN space (notably the `shape` branch) now carries the relational/local signal.
- **Per-prototype RDA covariance.** Mahalanobis covariance is now estimated per prototype from soft-assigned points, not shared per class.
- **Boundary geometry options.** `use_dual_boundary` (negative-face distances, 2F) and `boundary_init` (boundary-medoid prototype placement for `m>1`).

---

## Benchmarks

> **These numbers predate the Phase 9 two-space split and the removal of triangle scoring.** They reflect the earlier shared-vector / triangle-era architecture and should be re-run on the current build before being quoted. They are retained here only as historical reference. All results are 5-fold cross-validation macro F1; baselines use sklearn defaults with StandardScaler where required.

### Balanced Datasets *(historical)*

| Model | Iris | Wine (cultivar) | Digits |
|-------|------|-----------------|--------|
| LearningAVNN | **0.9598** | **0.9831** | **0.9844** |
| SVM-RBF | 0.9599 | 0.9834 | 0.9839 |
| Random Forest | 0.9464 | 0.9784 | 0.9788 |
| GBM | 0.9463 | 0.9515 | 0.9649 |
| XGBoost | 0.9456 | 0.9608 | 0.9621 |
| Logistic Reg | 0.9532 | 0.9829 | 0.9710 |

### Imbalanced Datasets — Synthetic *(historical)*

| Model | Synth-moderate (4/82/14%) | Synth-severe (60/25/10/5%) |
|-------|--------------------------|---------------------------|
| LearningAVNN | **0.6279** | **0.7574** |
| XGBoost | 0.6755 | 0.7787 |
| SVM-RBF | 0.5831 | **0.8039** |
| Random Forest | 0.6202 | 0.7401 |
| GBM | 0.6366 | 0.7441 |
| Logistic Reg | 0.5046 | 0.6052 |

Synth-moderate mirrors red wine (4% minority). Synth-severe mirrors ArrivalType (5% rarest class). SVM-RBF led on severe via implicit kernel interactions; XGBoost led on moderate via threshold splits; LearningAVNN was competitive with the tree ensembles and well ahead of the linear/kernel-free baselines.

### Architectural Comparisons *(still applicable)*

| Change | Impact |
|--------|--------|
| Full RDA Mahalanobis vs diagonal | +0.02–0.03 macro F1 on imbalanced |
| Independent sigmoid gates vs softmax | Lets `boundary` + `tanh_arccos` both be strong |
| Shape branch (z-score) vs no shape | Wine cultivar: 0.9884 → 0.9831 without it |
| Separate KNN space vs shared vector | Enables genuine AVM/KNN disagreement *(re-measure)* |

---

## Parameter Reference

### Core Training

| Parameter | Default | Notes |
|-----------|---------|-------|
| `k` | 5 | KNN neighbours |
| `lr` | 1e-3 | Adam learning rate |
| `epochs` | 300 | Max training epochs |
| `batch_size` | 256 | Mini-batch size |
| `patience` | 60 | Early stopping patience (in epochs) |
| `val_fraction` | 0.15 | Held-out fraction for early stopping + λ search |
| `random_state` | None | Reproducibility seed |
| `verbose` | False | Print training progress |

### Regularisation

| Parameter | Default | Notes |
|-----------|---------|-------|
| `label_smoothing` | 0.05 | NLL label smoothing |
| `weight_cap` | 1.5 | Max balanced class weight (prevents extreme minority over-weighting) |
| `weight_decay` | 1e-4 | Adam L2 regularisation |
| `feat_weight_reg` | 0.05 | Pulls per-feature weights toward uniform; use 0.01 for imbalanced |
| `ortho_reg` | 0.01 | Gram-matrix AVM-branch decorrelation penalty |

### Prototypes

| Parameter | Default | Notes |
|-----------|---------|-------|
| `centroid_reg` | 0.1 | Anchors prototypes near class means |
| `centroid_sep` | 0.05 | Inter-class prototype separation |
| `n_prototypes` | 1 | Prototypes per class; >1 for multimodal classes |
| `intra_sep` | 0.05 | Intra-class prototype spread (only active when `n_prototypes>1`) |
| `boundary_init` | False | Boundary-medoid prototype placement (only when `n_prototypes>1`) |

### Geometry / Distance

| Parameter | Default | Notes |
|-----------|---------|-------|
| `use_dual_boundary` | False | Extend the boundary branch to 2F with negative-face distances |
| `mahalanobis` | True | Full RDA Mahalanobis at inference (per-prototype covariance) |
| `mahal_reg` | 1e-6 | Ridge regularisation on covariance |
| `mahal_alpha` | 0.5 | QDA↔LDA blend; 0=per-class, 1=pooled. Use 0.2–0.3 for severe imbalance |

### Lambda Fusion

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lam_floor` | 0.5 | Minimum AVM weight in the scalar λ search |
| `entropy_lambda` | False | Use the per-sample entropy gate instead of auto scalar/confidence selection |
| `lam_confident` | 0.90 | λ when AVM is confident (low entropy) |
| `lam_uncertain` | 0.40 | λ when AVM is uncertain (high entropy) |

### Imbalance Handling

| Parameter | Default | Notes |
|-----------|---------|-------|
| `ordinal` | False | Ordinal EMD loss for ordered class labels |
| `ordinal_weight` | 0.5 | EMD loss blend weight |
| `supcon` | False | Supervised contrastive loss |
| `supcon_weight` | 0.3 | SupCon loss blend weight |
| `supcon_temp` | 0.1 | SupCon temperature |
| `ema_centroids` | False | EMA prototype stabilisation |
| `ema_beta` | 0.9 | EMA decay rate |
| `per_class_tau` | True | Per-class temperature scaling |
| `val_macro_bias` | 0.5 | Val metric: 0=½(acc+weighted), 1=macro F1 |

### FAISS

| Parameter | Default | Notes |
|-----------|---------|-------|
| `use_ivf` | True | IVF approximate index (≥78 training points); auto-falls back to FlatL2 |

### Fitted attributes (post-`fit`, for inspection)

`branch_weights_` (6: AVM tac/bnd/cir, then KNN lin/shp/sqd), `feat_weights_`, `tau_`, `lambda_` (`None` = a gate is in use), `n_prototypes_`, `best_val_f1_`, `centroid_drift_`, `centroid_drift_per_proto_`. Helpers: `summary()`, `class_distribution()`, `predict_with_uncertainty()`.

---

## Design Decisions and What Didn't Work

### Worked

**Separate AVM and KNN feature spaces** — When both branches scored the *same* combined vector, the late fusion was averaging two views of one geometry; they could never meaningfully disagree. Splitting them — a learned global space for AVM, a frozen local space for KNN — gives the fusion real signal. This is the central architecture of the current build.

**Independent sigmoid gates over softmax** — Softmax creates zero-sum competition; when `boundary` and `tanh_arccos` both need to be strong (as on imbalanced data), softmax forces them to fight. Normalised sigmoid resolves it.

**Per-prototype RDA covariance** — With `n_prototypes>1`, sharing one class-pooled covariance across prototypes contradicted spreading them with `intra_sep`. Estimating a covariance per prototype from soft-assigned points lets each sub-cluster have its own shape.

**Float64 Mahalanobis intermediates** — Float32 overflows on high-dimensional combined vectors → NaN → silent divide-by-zero. Promoting the matmul to float64 fixed it.

**val_macro_bias over three separate weights** — Collapsed a 3D correlated search space into one float, computed from a single confusion matrix (~4.5× faster than three sklearn metric calls).

### Didn't Work / Retired

**Triangle interaction scoring** — Real gains in its era, but removed in the current build; the dedicated KNN space (especially the `shape` branch) now supplies the relational/local signal it was capturing. *(Reason for removal not yet documented here — see Open Items.)*

**MLP lambda gate** — ~100 extra parameters and 100 epochs for no improvement over the zero-parameter entropy/confidence gates.

**Kernel trick (RBF triangle)** — Improved synth-severe by ~+0.025 but destroyed the interpretability of the triangle pairs; not worth the trade. (Moot now that triangle is gone.)

**Triplet scoring (3D cross-product)** — `C(F,3)` blowup (165 triplets at F=11, 1140 at F=20); gradient too diluted to find signal in 300 epochs.

**Bilinear W matrix** — `W²` init means zero gradient at `W=0`; even after fixing, it competed with triangle through shared centroids and hurt results.

**Branch pruning** — Never improved results; added ~80 lines of training-loop complexity.

---

## Open Items

| Item | Status | Notes |
|------|--------|-------|
| Re-run all benchmarks on the Phase 9 build | Pending | Current tables predate the two-space split and triangle removal; numbers must be regenerated before quoting |
| `n_prototypes > 1` | Implemented, needs validation | Per-prototype RDA covariance, soft-assignment EMA, and boundary-medoid init are all built; needs a clean ablation vs `n_prototypes=1` on the full datasets |
| `boundary_init` / `use_dual_boundary` | Implemented, untested | New geometry options; A/B vs defaults on ArrivalType / wine |
| Document why triangle was retired | Pending | Capture the decision (did the KNN split subsume it, or did it regress?) so the history is complete |
| Wine / ArrivalType benchmarks | Pending | Run `wine_arrival_test.py` on the full datasets and record real macro F1 |
| German Credit / Pima Diabetes | Pending network | Standard imbalanced benchmarks via `fetch_openml`; blocked by sandbox network |
| Vestigial `_lambda_gate` comment | Minor cleanup | Comment still references "AVM+tri" though triangle is gone |

---

## Files

| File | Purpose |
|------|---------|
| `LearningAVNN.py` | Model — import and use this |
| `auto_lambda_test.py` | Full benchmark: Iris, Wine, Breast Cancer, Red Wine, White Wine, ArrivalType |
| `quick_test.py` | Fast benchmark: Iris, Wine, Digits, Synth-moderate, Synth-severe |
| `wine_arrival_test.py` | Red Wine, White Wine, ArrivalType |
| `benchmark_all_models.py` | LearningAVNN vs XGBoost, LightGBM, SVM, RF, GBM |
| `requirements.txt` | `torch`, `numpy`, `scikit-learn`, `pandas`, `faiss-cpu` |
| `ARCHITECTURE.md` | Detailed architecture reference |
