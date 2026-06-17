# LearningAVNN — Architecture Reference

**sklearn-compatible · two feature spaces · QDA-scored geometric classifier**

---

## Core Concept

A geometric distance classifier that learns *what kind of distance* matters for a
given dataset — not just how far a point is from a class prototype, but the shape
and orientation of each class's cluster (via a learned per-prototype covariance)
and which geometric "views" of the feature vector are informative.

It never learns a decision boundary directly. It learns a **metric** (feature
weights + per-class precision matrices) and a set of **landmarks** (class
prototypes); the boundary is the equal-distance locus between landmarks under
that metric.

---

## Pipeline

```
Raw features X
    → MinMax normalise to [-1, 1]                 (fit on training data only)
    → TWO separate feature maps:
        AVM space (learned, 3 branches, 3F or 4F wide)
        KNN space (frozen, configurable, N·F wide)
    → AVM head : QDA posterior over K×m learnable prototypes  ─┐
    → KNN head : FAISS nearest-neighbour vote                  ─┤
    → (optional) Fisher head : LDA posterior                   ─┤ → fusion → prediction
    → (optional) geodesic head : Isomap nearest-centroid       ─┘
```

Two distinct spaces, not one combined vector: the AVM space is gradient-trained
and carries a covariance metric; the KNN space is frozen and Euclidean. Extra
geometric views (log/rank/clr/interaction) live in the KNN space precisely
because it pays no per-prototype covariance dimensionality tax.

---

## The AVM space (learned, global structure)

Three branches, each independent-sigmoid-gated and normalised to sum to 1. `tac`
and `bnd` start at sigmoid(0.5) ≈ 0.62 vs `cir` at sigmoid(0) = 0.50 — empirically
dominant branches get a head start. Per-feature weights `fw` (a diagonal metric)
are learned alongside.

| Branch | Formula | Signal |
|--------|---------|--------|
| tanh_arccos (tac) | `tanh(arccos(x)·0.8)·√fw` | Monotone angular position, centre-amplified |
| boundary (bnd) | `√(‖x‖²−2x+1) = ‖x−eᵢ‖` | Cross-feature coupling via ‖x‖² (the only non-separable branch; **not** feature-weighted) |
| circular (cir) | `√(1−x²)·√fw = ‖sinθ‖·√fw` | Symmetric extremeness |

`use_dual_boundary` adds the opposite faces `‖x+eᵢ‖`, interleaved, giving
distances to all 2F cube vertices (AVM dimension 4F).

## The KNN space (frozen, local structure)

Configurable via `knn_branches`, default `('linear','shape','quadratic')`,
gradient-free, uniformly weighted, fed to a FAISS index.

| Branch | Formula | Signal |
|--------|---------|--------|
| linear | `x·√fw` | Signed position + magnitude |
| shape | `(x−μ)/σ` per sample | Within-sample relative profile (scale/level invariant) |
| quadratic | `x²·√fw` | Nonlinear extremeness |
| log *(opt)* | `sign(x)·log(1+\|x\|)·√fw` | Heavy-tail / multiplicative warp |
| rank *(opt)* | train-ECDF of x | Monotone, scale-free reparametrisation |
| clr *(opt)* | `log x − mean log x` | Compositional (simplex) geometry |
| interaction *(opt)* | random-projected pairwise products xᵢ·xⱼ, z-scored | Cross-feature nonlinear probe (`interaction_dim` wide) |

---

## Learnable Prototypes

`nn.Parameter` of shape `(K, m, F)` in raw normalised space; moved by backprop +
optional EMA stabilisation.

- `n_prototypes=1` — single centroid per class (default).
- `n_prototypes>1` — a mixture; `intra_sep` spreads sub-centroids to cover
  sub-modes, and each gets its own covariance via soft responsibilities. Class
  regions become piecewise-quadric / possibly non-convex.
- `boundary_init` (m>1) — prototype 0 at the class centroid, the rest at
  lowest-positive-margin "frontier" training points.

EMA writes reset the optimiser's momentum state for the centroid parameter so
stale Adam moments don't fight the discontinuous EMA move.

**Prototype potentials:** anchor (`centroid_reg`, spring to init means),
inter-class repulsion (`centroid_sep`), intra-class repulsion (`intra_sep`,
m>1). The separation terms stay attached to the autograd graph (detaching them
silently disables them — a fixed bug).

---

## AVM Scoring — QDA posterior

**Training** scores prototypes with Euclidean inverse-distance in the AVM image
space (cheap, differentiable). **Inference** re-scores with the QDA posterior,
selected by `avm_score`:

`avm_score='qda'` (default) — the Bayes-form posterior, equal-weight mixture over
a class's m prototypes:

```
log p(k|x) = prior_weight·log π_k
           + logsumexp_{p∈k}[ −0.5·(logdet_weight·log|Σ_p| + d²_M(x, μ_p)) ]
```

then softmax over k. `d²_M` is the per-prototype Mahalanobis distance.

`avm_score='inverse_distance'` — the original harmonic vote
`Σ_p 1/(d_M/τ + ε)`, renormalised. Flat, heavy-tailed, better-calibrated for the
downstream entropy gates; `logdet_weight` has no analogue here and is ignored.

Dials: `logdet_weight` scales just the covariance-volume term; `prior_weight`
scales the class prior in both modes (0 → no prior → higher minority recall).
The pre-QDA behaviour is exactly `avm_score='inverse_distance', prior_weight=0`.

### RDA Mahalanobis covariance

Per-prototype, estimated on **per-column-standardised** combined features (so the
ridge and pooling aren't dominated by the large-scale boundary block):

```
Σ_p = (1−α)·S_p + α·S_pooled        # α = mahal_alpha: 0 = QDA, 1 = LDA
Σ_p ← (1−r)·Σ_p + r·σ̄²·I            # r = mahal_reg ridge
```

Inverted by Cholesky in float64 with an escalating-ridge retry (×10, up to 3×)
before falling back to pseudo-inverse; `log|Σ_p|` is read off the Cholesky factor
for the QDA volume term. For m>1, each prototype's covariance is estimated from
its soft-assigned points with an effective-sample-size (weighted Bessel)
correction.

> Triangle and bilinear interaction scoring (in older docs) were **removed**: the
> per-prototype precision matrix's off-diagonal entries are exactly the learned
> pairwise-interaction terms `(Σ⁻¹)ᵢⱼ(xᵢ−μᵢ)(xⱼ−μⱼ)`, so explicit degree-2
> scoring was redundant. Higher-order structure comes from the prototype mixture
> or a future learned projection.

---

## Heads & Fusion

| Head | What it is |
|------|-----------|
| `avm` | QDA posterior over learnable prototypes (above) |
| `knn` | FAISS inverse-distance vote in the KNN space (IVF, flat fallback) |
| `fisher` | LDA (eigen solver, Ledoit-Wolf shrinkage) — global discriminative directions |
| `geodesic` | Isomap unfolding + nearest-centroid (least-validated, off by default) |

`heads=('avm','knn')` by default. Fusion modes, chosen on the validation metric:

- **avm_only** — single head.
- **legacy λ** (avm+knn) — `λ·avm + (1−λ)·knn`; λ scalar-grid-searched, per-sample
  **confidence gate** (entropy-based), or **entropy gate**.
- **N-head** (any larger set) — convex combination; **confidence** (per-sample,
  1−entropy) or **static** (coordinate-ascent on val, shrunk toward uniform with a
  per-head floor so no head is zeroed). Selection prefers uniform ≺ confidence ≺
  static, moving up only past a margin.

`prior_temp` is a separate, optional **post-fusion** prior gravity
`p' ∝ p·π^α` — distinct from the in-discriminant `prior_weight`; default 0.

---

## Training Losses

| Loss | Purpose | Controlled by |
|------|---------|---------------|
| Weighted NLL + label smoothing | Classification, imbalance | `weight_cap`, `label_smoothing` |
| Ordinal EMD | Penalise ordinal mistakes proportionally (1-Wasserstein on the label line) | `ordinal`, `ordinal_weight` |
| Supervised Contrastive | Pull same-class AVM embeddings together on the sphere | `supcon`, `supcon_weight`, `supcon_temp` |
| Branch orthogonality | Decorrelate the AVM branches (Gram → I; dual-boundary aware) | `ortho_reg` |
| Centroid anchor / sep / intra | Prototype potential field | `centroid_reg`, `centroid_sep`, `intra_sep` |
| Feature-weight spring | Pull the diagonal metric toward isotropy | `feat_weight_reg` |

Adam under CosineAnnealingWarmRestarts (T₀=30, T_mult=2), unit-norm gradient
clipping, early stopping on the composite validation metric `V` after warmup.

---

## Inference

```python
predict_proba(X):
    Xn       = normalise(X)                      # train min/range
    avm      = QDA posterior (Mahalanobis) over K×m prototypes   # or inv-distance
    knn      = FAISS KNN vote in the KNN space
    fused    = fusion(avm, knn, [fisher, geodesic])
    return   prior_temp gravity (optional) applied to fused
```

## Inspection

- `summary(feature_names=None)` — learned branch/feature weights, τ, fusion,
  centroid drift, lambda.
- `report(X, y)` — per-class precision/recall/F1/support + confusion matrix
  (flags the lowest-F1 class); `as_dict=True` for programmatic sweeps.
- `predict_with_uncertainty(X)` — predictions, probabilities, per-sample head
  disagreement.
- `class_distribution()`, and fitted attributes: `branch_weights_`,
  `feat_weights_`, `tau_`, `centroid_drift_`, `cov_logdet_`, `fusion_`.

---

## Open Items

| Item | Status |
|------|--------|
| Free-coordinate interaction prototypes (AVM space) | Sketched, not integrated |
| Bandit / sparse-projection interaction selection | Designed (pairs with free coords) |
| Learned low-rank projection `L` (feature-combination metric) | Proposed (core_math §15) |
| GPU path | CPU-only today |

---

## Parameter Reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| `k` | 5 | KNN neighbours |
| `lr` | 1e-3 | Adam learning rate |
| `epochs` | 300 | Max training epochs |
| `batch_size` | 256 | Mini-batch size |
| `patience` | 60 | Early-stopping patience (epochs) |
| `val_fraction` | 0.15 | Internal val split (early stop + λ/fusion select) |
| `label_smoothing` | 0.05 | NLL label smoothing |
| `weight_cap` | 1.5 | Max balanced class weight |
| `weight_decay` | 1e-4 | Adam L2 |
| `feat_weight_reg` | 0.05 | Per-feature weight spring toward isotropy |
| `centroid_reg` | 0.1 | Prototype anchor strength |
| `centroid_sep` | 0.05 | Inter-class prototype repulsion |
| `n_prototypes` | 1 | Prototypes per class |
| `intra_sep` | 0.05 | Intra-class prototype repulsion (m>1) |
| `boundary_init` | False | Boundary-medoid prototype placement (m>1) |
| `use_dual_boundary` | False | Boundary branch → 2F (both cube faces) |
| `ortho_reg` | 0.01 | Branch decorrelation |
| `mahalanobis` | True | Full RDA covariance at inference |
| `mahal_reg` | 1e-6 | Covariance ridge |
| `mahal_alpha` | 0.5 | QDA(0) ↔ LDA(1) blend |
| `logdet_weight` | 1.0 | Scales the QDA volume term −½·log\|Σ\| |
| `avm_score` | `'qda'` | `'qda'` or `'inverse_distance'` |
| `prior_weight` | 1.0 | Scales the class prior π_k in the score |
| `lam_floor` | 0.5 | Min AVM weight in λ grid |
| `entropy_lambda` | False | Per-sample λ gate |
| `lam_confident` | 0.90 | λ when AVM entropy low |
| `lam_uncertain` | 0.40 | λ when AVM entropy high |
| `ordinal` | False | EMD ordinal loss |
| `ordinal_weight` | 0.5 | EMD blend |
| `per_class_tau` | True | Per-class training temperature |
| `prior_temp` | 0.0 | Post-fusion prior gravity (separate from prior_weight) |
| `val_macro_bias` | 0.5 | 0 = accuracy, 1 = macro-F1 for val scoring |
| `use_ivf` | True | FAISS IVF (approximate, fast) |
| `supcon` | False | Supervised contrastive loss |
| `supcon_weight` | 0.3 | SupCon blend |
| `supcon_temp` | 0.1 | SupCon temperature |
| `ema_centroids` | False | EMA prototype stabilisation |
| `ema_beta` | 0.9 | EMA decay |
| `heads` | `('avm','knn')` | Active heads |
| `knn_branches` | `('linear','shape','quadratic')` | KNN-space views |
| `interaction_dim` | `'auto'` | Width of the `interaction` branch (auto = n_features) |
| `geodesic_neighbors` | 10 | Isomap k (geodesic head) |
| `geodesic_components` | 5 | Isomap output dims |
| `geodesic_max_fit` | 4000 | Isomap fit subsample cap |
| `device` | `'auto'` | CUDA if available else CPU (training loop only) |
| `random_state` | None | Reproducibility seed |
| `verbose` | False | Training progress output |
