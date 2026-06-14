# LearningAVNN

**A geometric tabular classifier with learnable distance metrics, separate global/local feature spaces, configurable fusion heads, and imbalance handling.**

`sklearn`-compatible · pure PyTorch training · FAISS inference · 43 parameters · ~1580 lines

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
9. [Files](#files)

---

## What It Is

LearningAVNN (Learning Angular Vector Neural Network) is a geometric distance-based tabular classifier. Instead of learning a standard decision boundary, it learns *what kind of distance* matters — transforming features into geometric spaces, then measuring how close a sample is to learned class prototypes.

The core is a two-space split:

- A **learned AVM space** (global structure) scores each sample by its per-prototype Mahalanobis proximity to `K×m` learnable class prototypes.
- A **separate, frozen KNN space** (local structure) scores each sample by an inverse-distance vote over its nearest training neighbours via FAISS.

Those are two of the available **fusion heads**. The head set is configurable (`heads`): the default `('avm','knn')` reproduces the original late-fusion `λ·AVM + (1−λ)·KNN` exactly, and you can add a **Fisher** head (an LDA-direction posterior) and a **geodesic** head (an Isomap manifold embedding) when a dataset rewards them. With more than the two legacy heads, an N-head fusion gate blends them, choosing on validation between regularised static weights and a per-sample confidence gate.

Because the spaces use different branch transforms, they genuinely disagree — the fusion can correct one head with another rather than averaging two views of the same vector. It was designed for imbalanced multi-class tabular classification, where tree ensembles and kernel methods struggle to recover rare-class signal without sacrificing majority-class accuracy.

---

## Quick Start

```python
from LearningAVNN import LearningAVNN

# Balanced dataset (legacy AVM+KNN path)
model = LearningAVNN(k=5, epochs=300)

# Balanced/clean, want a little extra: add the Fisher head
model = LearningAVNN(k=5, epochs=300, heads=('avm', 'knn', 'fisher'))

# Imbalanced dataset — enrich the KNN space with monotone views
model = LearningAVNN(
    ordinal=True, supcon=True, ema_centroids=True,
    val_macro_bias=0.8,
    knn_branches=('linear', 'shape', 'quadratic', 'log', 'rank'),
)

# Severe imbalance + multimodal classes (e.g. Red Wine 4/82/14):
#   keep AVM+KNN, enrich the KNN branches, DO NOT add Fisher (it dilutes here)
model = LearningAVNN(
    n_prototypes=2, ema_centroids=True, supcon=True, ordinal=True,
    val_macro_bias=0.8, mahal_alpha=0.3,
    heads=('avm', 'knn'),
    knn_branches=('linear', 'shape', 'quadratic', 'log', 'rank'),
)

# ArrivalType — reference config (oil & gas, ~2.17M rows, ~5% rarest class)
model = LearningAVNN(
    n_prototypes=2,            # ArrivalType classes are multimodal
    ema_centroids=True,        # stabilise minority prototypes under sparse gradient
    mahal_alpha=0.3,           # lean toward per-class QDA for severe imbalance
    weight_cap=4.0,            # allow stronger minority up-weighting at this scale
    val_macro_bias=0.85,       # tune toward macro F1, not accuracy
    k=10, use_ivf=True,        # IVF index for the 2.17M-row training set
    heads=('avm', 'knn', 'fisher'),
    knn_branches=('linear', 'shape', 'quadratic', 'log', 'rank'),
)

model.fit(X_train, y_train)
pred  = model.predict(X_test)
proba = model.predict_proba(X_test)
print(model.summary(feature_names))

# Optional: per-sample disagreement between heads
pred, proba, disagreement = model.predict_with_uncertainty(X_test)
```

> **Head selection is dataset-dependent and worth screening.** Fisher helps on clean/moderate sets (Breast Cancer, White Wine) but *dilutes* on multimodal severe imbalance (Red Wine: 0.59 → 0.53 even after the fusion was made robust), because a global-linear discriminant is the wrong shape for that regime. Enriched KNN branches (`log`, `rank`) help imbalanced sets broadly. Use `benchmark_heads.py` to compare variants per dataset before committing. `device='auto'` uses CUDA when available.

---

## Architecture

### Pipeline

```
Raw X  →  MinMax normalise [-1,1]
       →  Heads (configurable via `heads`):
            avm   : learned AVM space  (tac + boundary[·dual] + circular)
                       → per-prototype Mahalanobis inverse-distance to K×m prototypes
            knn   : frozen KNN space   (linear + shape + quadratic [+ log/rank/clr])
                       → FAISS inverse-distance vote over k nearest training points
            fisher: LDA(eigen, shrinkage='auto') posterior on the normalised space   (optional)
            geo   : Isomap embedding → nearest-class-centroid scoring                 (optional)
       →  Fusion:
            legacy (avm,knn)     : λ·AVM + (1−λ)·KNN   (scalar / confidence / entropy gate)
            N-head (any other set): regularised static weights OR confidence gate,
                                    auto-selected on validation
       →  optional decision-stage class-prior reweight (`prior_temp`, default off)
```

The AVM and KNN spaces are **distinct**. Only the AVM space participates in the training forward pass and receives gradient; the KNN branch weights are parameters but get no gradient, so they remain a fixed geometric view that complements the learned space. Per-feature weights `fw` (softmax-normalised to sum to `n_feat`, so uniform = 1.0 each) are learned once and shared across both spaces.

### AVM Branches — global structure (learned)

Three branches, combined into a `3F`-wide vector (`4F` if `use_dual_boundary`). Branch weights are independent normalised-sigmoid gates (not softmax), so multiple branches can be strong at once. `tanh_arccos` and `boundary` are initialised with a `+0.5` raw-logit head start over `circular`.

| Branch | Formula | Signal |
|--------|---------|--------|
| tanh_arccos | `tanh(arccos(x)·0.8)·√fw` | Monotone nonlinear; high at x≈−1, ~0 at x≈+1; most sensitive near x=0 |
| boundary | `√(‖x‖²−2x+1)` | Cross-feature coupling via the global term ‖x‖² |
| circular | `√(1−x²)·√fw` | Symmetric extremeness; peaks at ±1 |

**`use_dual_boundary`** (opt-in) extends the boundary branch to `2F` by adding the negative-face distances `√(‖x‖²+2x+1)`, **interleaved** with the positive-face distances as `[pos_f0, neg_f0, pos_f1, neg_f1, …]`. (The numpy inference path and the torch training path now build this identical interleaved layout — an earlier mismatch built one as a block-concat and the other interleaved, silently scrambling the axes between train and inference.)

### KNN Branches — local structure (frozen, configurable)

Combined into a separate vector that feeds the FAISS index. These weights receive no gradient and stay uniform. The set is configurable via `knn_branches`; the first three are the default, the rest are opt-in monotone/compositional views that help on imbalanced data.

| Branch | Formula | Signal |
|--------|---------|--------|
| linear | `x·√fw` | Signed magnitude; direct local proximity |
| shape | `(x−μ)/σ` per sample | Within-sample relative profile fingerprint |
| quadratic | `x²·√fw` | Nonlinear extremeness |
| log *(opt-in)* | signed `log1p` | Multiplicative geometry; compresses heavy tails |
| rank *(opt-in)* | empirical-quantile ECDF (fit on train) | Monotone, scale-free; robust to skew |
| clr *(opt-in)* | centered log-ratio (offset fit on train) | Compositional features only — use with care |

Why these live in the KNN space and not the AVM space: they are uniform-weighted in a flat Euclidean vote, so they pay no covariance/dimensionality tax. Augmenting the *AVM* vector with them instead wrecks it (covariance blow-up, transform mismatch, ‖x‖² pollution) — a view only helps as its own late-fused contribution, which is exactly why the head/branch design exists.

### Fusion Heads (optional)

| Head | What it is | Notes |
|------|-----------|-------|
| `avm` | learned Mahalanobis-to-prototypes (above) | core head |
| `knn` | frozen FAISS inverse-distance vote (above) | core head |
| `fisher` | `LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')` posterior on the normalised features | its own LDA decision rule, not run through AVM branches; strong on clean/moderate, **weak on multimodal severe imbalance** |
| `geodesic` | Isomap embedding (subsampled to `geodesic_max_fit`) + nearest-class-centroid scoring | heaviest and least validated; off by default |

### N-Head Fusion (robust selector)

For the legacy `('avm','knn')` set, fusion is the original λ blend (below). For any other head set, `fit` calls `_fit_fusion` on the validation split and picks among:

- **uniform** weights (the anchor),
- a **confidence gate** (per-sample, parameter-free), and
- **static** coordinate-ascent weights.

The static search is **regularised against overfitting a small/imbalanced validation split**: every head weight is **floored** (no head can be zeroed), the fitted weights are **shrunk 50% toward uniform**, and the selector only departs from uniform when the gain clears a margin, preferring `uniform < confidence < static`. This was added after the unregularised selector zeroed the AVM head on Red Wine's ~190-point val split (4% class ≈ 7 samples) and collapsed macro-F1 to 0.49. The selector now fails safe.

### Learnable Prototypes

Prototypes are an `nn.Parameter` of shape `(K, m, F)` — `K` classes, `m` prototypes each — updated by backprop plus optional EMA stabilisation. `centroid_reg` anchors them near their initial class means; `centroid_sep` pushes prototypes of different classes apart; `intra_sep` pushes the prototypes *within* a class apart (active only when `m>1`).

> **Correctness fix:** the separation losses (`centroid_sep`, `intra_sep`) were previously computed `.detach()`ed and cached once per epoch, so they contributed **zero gradient** — the knobs were inert. They are now recomputed attached each batch and are **live**. Consequence: the `intra_sep=0.05` default now actively opposes EMA on multi-prototype configs and should be re-swept (see `sweep_intra_sep.py`); `intra_sep=0` is often best for `n_prototypes>1`.

Two initialisation modes:

- **Jitter init (default)** — every prototype starts at its class mean; for `m>1`, small Gaussian jitter breaks symmetry so the NLL gradient and `intra_sep` can spread them.
- **Boundary-medoid init (`boundary_init=True`, only when `m>1`)** — prototype 0 sits at the class centroid (interior anchor); prototypes 1…m−1 are placed at the lowest-positive-margin training points (frontier defenders), where margin = `d(nearest opposing centroid) − d(own centroid)`. Negative-margin points are excluded as likely label noise; degenerate classes fall back to jitter. (Placement now uses a norm-expansion matmul rather than an `(N,K,F)` broadcast, to bound memory.)

### AVM Scoring

For each prototype, score is an inverse distance; per-class score sums over that class's `m` prototypes, then normalises:

$$\text{score}_k(x) = \sum_{j=1}^{m} \frac{1}{d(x, \mu_{kj}) + \varepsilon}$$

In the training forward pass, `d` is Euclidean in the AVM space and a learned per-class temperature `τ` (when `per_class_tau=True`) lets minority classes adopt softer scoring.

At inference, `d` is a **full RDA Mahalanobis** distance with a per-prototype covariance:

$$\Sigma_{kj} = (1-\alpha)\,S_{kj} + \alpha\,S_{\text{pooled}}, \qquad \Sigma_{kj}^{\text{reg}} = (1-r)\,\Sigma_{kj} + r\cdot\overline{\text{diag}}\cdot I$$

`α` (`mahal_alpha`) blends per-class QDA (`α→0`) toward pooled LDA (`α→1`); `r` (`mahal_reg`) is the ridge that keeps each matrix positive-definite when `n_k < D`. Inversion uses Cholesky with an **escalating-ridge retry** (×10, up to 3×) before falling back to a pseudo-inverse, and intermediates are held in float64 (float32 overflows on high-dimensional combined vectors → NaN → silent divide-by-zero). When `m>1`, each prototype gets its own covariance estimated from the training points soft-assigned to it. The per-row Mahalanobis pass is **batched** (`O(batch·D)` memory, not `O(N·D)`) so it scales to ArrivalType-size data.

### Lambda Fusion (legacy `avm`+`knn` path)

The fused posterior is `λ·AVM + (1−λ)·KNN`. Three λ regimes:

| Mode | Behaviour |
|------|-----------|
| Scalar (auto) | Grid-searched on `linspace(lam_floor, 1.0)`; `lam=1.0` (pure AVM) is always a candidate |
| Confidence gate (auto) | Per-sample `λ = c_AVM / (c_AVM + c_KNN)`, `c = 1 − normalised entropy`, clipped to `[lam_uncertain, lam_confident]` |
| Entropy gate (`entropy_lambda=True`) | Per-sample `λ(x) = lam_confident − (lam_confident − lam_uncertain)·H(x)` |

When `entropy_lambda=False`, `fit` evaluates both the scalar search and the confidence gate on validation and keeps the better. If FAISS is unavailable the model falls back to pure AVM (`λ=1.0`).

### Centroid Gravity (`prior_temp`, decision-stage, default off)

An optional class-prior reweight applied **after** fusion, **before** the decision:

$$p'(k\mid x) \propto p(k\mid x)\cdot \pi_k^{\alpha}, \qquad \alpha = \texttt{prior\_temp}$$

`α=0` (default) is no gravity — unchanged behaviour; `α=1` is the full empirical prior (the Bayes/accuracy pull toward the majority class); between is tempered. It is applied at the decision stage so it reshapes the decision without corrupting head likelihoods, and so a single fit can be swept over `α`.

**Empirical finding (see `sweep_prior_temp.py`):** on severe imbalance, static gravity is *strictly dominated* — `α=0` is the peak of the accuracy↔macro-F1 frontier in **both** directions. Positive `α` doesn't even buy accuracy (the heads already beat the majority-rate baseline, so gravity only erases the minority); negative `α` (minority up-weighting) also degrades both axes. The mechanism: the fused posterior is low-dynamic-range (normalised inverse-distance), so the ~3-nat prior odds of an 80/4 split dominate the likelihood at any meaningful `α`. The productive lever is `per_class_tau` (decision-shaping at training time on uncorrupted probabilities), not a post-hoc prior. The knob is retained at `0.0` for experimentation (e.g. ArrivalType, whose calibration may differ).

### Inference

```python
predict_proba(X):
    Xn   = normalise(X)                       # train statistics
    if legacy (avm,knn):
        avm = Mahalanobis inverse-distance     # to K×m prototypes
        if no FAISS index or λ≥1.0: p = avm
        else: knn = FAISS vote; p = λ·avm + (1-λ)·knn
    else:
        p = fuse(collect head probabilities)   # N-head static / confidence
    return apply_gravity(p)                     # no-op when prior_temp == 0
```

`predict_with_uncertainty(X)` additionally returns a per-sample disagreement score (heads confident but predicting different classes) — useful for triage and active labelling.

---

## Evolution History

### Phases 1–8 — Foundation through Cleanup

Started as a static three-branch FAISS classifier (`FastTriBranchAVNN`); added learnable branch/feature weights and **independent sigmoid gates** (over softmax); upgraded to **full RDA Mahalanobis** (Cholesky, float64); added the imbalance toolkit (SupCon, EMA centroids, ordinal EMD, per-class τ); extended centroids to a **mixture of `m` prototypes**; trialled and then **retired** triangle interaction scoring, kernel/triplet/bilinear variants, MLP λ gate, and branch pruning; consolidated three validation weights into `val_macro_bias`.

### Phase 9 — Two-Space Split

AVM and KNN stopped sharing one combined vector: AVM gets `tanh_arccos + boundary + circular` (learned), KNN gets `linear + shape + quadratic` (frozen). Triangle scoring removed (the KNN `shape` branch and per-prototype Mahalanobis now carry the relational/local signal). Per-prototype RDA covariance; `use_dual_boundary` and `boundary_init` geometry options added.

### Phase 10 — Correctness Pass, Configurable Heads & Branches *(current build)*

- **Correctness fixes** (all unit-tested in `test_avnn.py`): the separation-loss zero-gradient bug (knobs were inert, now live); batched `_avm_proba` (memory-bounded at scale); dual-boundary numpy/torch layout unified to interleave; escalating-ridge Cholesky retry in covariance inversion; boundary-init memory fix; GPU support via `device='auto'`.
- **Configurable fusion heads** (`heads`): added the **Fisher** (LDA-direction) and **geodesic** (Isomap) heads alongside `avm`/`knn`. The legacy `('avm','knn')` path is byte-for-byte unchanged.
- **Configurable KNN branches** (`knn_branches`): added `log`, `rank`, `clr` monotone/compositional views.
- **N-head fusion** with a **robust selector** (floor + shrink-to-uniform + margin) after the unregularised version overfit Red Wine's tiny imbalanced val split and zeroed the AVM head.
- **Centroid gravity** (`prior_temp`) added as a decision-stage knob and **measured to be strictly dominated** on severe imbalance — kept at default-off.
- **Benchmarks regenerated** on the current build, with a same-protocol baseline comparison (`baselines.py`).

---

## Benchmarks

All results are 5-fold stratified cross-validation, `random_state=42`, macro F1. These reflect the **current build**.

### LearningAVNN (current)

| Dataset | Accuracy | Macro F1 | Config notes |
|---------|----------|----------|--------------|
| Iris | 0.9600 | 0.9598 | heads = avm/knn/fisher |
| Wine (cultivar) | 0.9830 | 0.9831 | heads = avm/knn/fisher |
| Breast Cancer | 0.9631 | 0.9597 | + fisher, enriched KNN, confidence fusion |
| Red Wine (4/82/14) | 0.8580 | 0.5904 | **avm/knn**, enriched KNN (Fisher dropped — see below) |
| White Wine | 0.6952 | 0.6916 | + fisher, enriched KNN, confidence fusion |
| ArrivalType | — | *pending* | run locally; CSV not available in the build sandbox |

### vs standard baselines (same 5-fold protocol, macro F1)

`class_weight='balanced'` on SVM/RF; `StandardScaler` on distance/linear models.

| Dataset | Leaders | LearningAVNN | Standing |
|---------|---------|--------------|----------|
| Iris | KNN / LDA 0.973 · SVM 0.960 | **0.960** | mid-pack; tied SVM / RF |
| Wine | RF 0.984 · SVM 0.983 · LDA 0.983 | **0.983** | top cluster (5 within 0.001) |
| Breast | SVM 0.972 · LogReg 0.971 | **0.960** | mid-pack; tied KNN, ahead of RF/LDA/NB |

On clean, near-linearly-separable benchmarks LearningAVNN is competitive mid-pack — tied with the leading cluster on Wine, tied with SVM/RF on Iris, behind SVM-RBF on Breast — and consistently beats gradient boosting (HistGBM) on all three. These are not its design target; imbalanced multiclass is. The meaningful comparison is Red/White wine and ArrivalType — run `baselines.py` (it mirrors `benchmark.py`'s binnings/CV) to get those rows on real data.

### The Fisher-on-Red-Wine ablation

A clean three-rung result that drove the head-selection guidance:

| Red Wine config | Macro F1 | What happened |
|-----------------|----------|----------------|
| broken fusion + Fisher | 0.4945 | AVM head zeroed (the val-overfitting bug) |
| robust fusion + Fisher | 0.5291 | AVM safe, but Fisher's ~0.35 weight dilutes |
| robust fusion, avm/knn | **0.5904** | best — Fisher dropped |

Both interventions were needed: the fusion fix to stop the collapse, *and* dropping Fisher to stop the dilution.

### Architectural comparisons *(still applicable)*

| Change | Impact |
|--------|--------|
| Full RDA Mahalanobis vs diagonal | +0.02–0.03 macro F1 on imbalanced |
| Independent sigmoid gates vs softmax | Lets `boundary` + `tanh_arccos` both be strong |
| Enriched KNN branches (`log`,`rank`) | Breast +0.008, synth-severe +0.032 |
| Fisher head | Breast +0.023, but Red Wine −0.06 (multimodal severe) |
| Separate KNN space vs shared vector | Enables genuine AVM/KNN disagreement |

---

## Parameter Reference

### Core Training

| Parameter | Default | Notes |
|-----------|---------|-------|
| `k` | 5 | KNN neighbours |
| `lr` | 1e-3 | Adam learning rate |
| `epochs` | 300 | Max training epochs |
| `batch_size` | 256 | Mini-batch size |
| `patience` | 60 | Early-stopping patience (epochs) |
| `val_fraction` | 0.15 | Held-out fraction for early stopping + fusion search |
| `device` | `'auto'` | `'auto'`/`'cpu'`/`'cuda'`; auto uses CUDA when present |
| `random_state` | None | Reproducibility seed |
| `verbose` | False | Print training progress |

### Regularisation

| Parameter | Default | Notes |
|-----------|---------|-------|
| `label_smoothing` | 0.05 | NLL label smoothing |
| `weight_cap` | 1.5 | Max balanced class weight (4.0 at ArrivalType scale) |
| `weight_decay` | 1e-4 | Adam L2 |
| `feat_weight_reg` | 0.05 | Pull feature weights toward uniform; 0.01 for imbalanced |
| `ortho_reg` | 0.01 | Gram-matrix AVM-branch decorrelation |

### Prototypes

| Parameter | Default | Notes |
|-----------|---------|-------|
| `centroid_reg` | 0.1 | Anchors prototypes near class means |
| `centroid_sep` | 0.05 | Inter-class prototype separation (now a **live** gradient) |
| `n_prototypes` | 1 | Prototypes per class; >1 for multimodal classes |
| `intra_sep` | 0.05 | Intra-class spread (`m>1` only; now **live** — re-sweep, often 0 is best) |
| `boundary_init` | False | Boundary-medoid placement (`m>1` only) |

### Geometry / Distance

| Parameter | Default | Notes |
|-----------|---------|-------|
| `use_dual_boundary` | False | Extend boundary branch to 2F with negative-face distances |
| `mahalanobis` | True | Full RDA Mahalanobis at inference (per-prototype covariance) |
| `mahal_reg` | 1e-6 | Ridge on covariance (with escalating-retry Cholesky) |
| `mahal_alpha` | 0.5 | QDA↔LDA blend; 0.2–0.3 for severe imbalance |

### Heads & Branches

| Parameter | Default | Notes |
|-----------|---------|-------|
| `heads` | `('avm','knn')` | Add `'fisher'` and/or `'geodesic'`; default = legacy path |
| `knn_branches` | `('linear','shape','quadratic')` | Add `'log'`,`'rank'` for imbalanced; `'clr'` only if compositional |
| `geodesic_neighbors` | 10 | Isomap neighbours (geodesic head) |
| `geodesic_components` | 5 | Isomap output dimensionality |
| `geodesic_max_fit` | 4000 | Subsample cap for fitting Isomap |

### Lambda Fusion (legacy path)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lam_floor` | 0.5 | Minimum AVM weight in the scalar λ search |
| `entropy_lambda` | False | Per-sample entropy gate instead of auto scalar/confidence |
| `lam_confident` | 0.90 | λ when AVM is confident |
| `lam_uncertain` | 0.40 | λ when AVM is uncertain |

### Imbalance / Decision

| Parameter | Default | Notes |
|-----------|---------|-------|
| `ordinal` | False | Ordinal EMD loss for ordered labels |
| `ordinal_weight` | 0.5 | EMD blend weight |
| `supcon` | False | Supervised contrastive loss |
| `supcon_weight` | 0.3 | SupCon blend weight |
| `supcon_temp` | 0.1 | SupCon temperature |
| `ema_centroids` | False | EMA prototype stabilisation |
| `ema_beta` | 0.9 | EMA decay |
| `per_class_tau` | True | Per-class temperature (the productive decision-shaping knob) |
| `prior_temp` | 0.0 | Decision-stage class-prior reweight ('gravity'); 0 = off |
| `val_macro_bias` | 0.5 | Val metric: 0 = ½(acc+weighted), 1 = macro F1 |

### FAISS

| Parameter | Default | Notes |
|-----------|---------|-------|
| `use_ivf` | True | IVF approximate index (≥78 points); auto-falls back to FlatL2 |

### Fitted attributes (post-`fit`, for inspection)

`branch_weights_`, `feat_weights_`, `tau_`, `lambda_` (`None` = a gate is in use), `fusion_` (mode + heads + weights for N-head sets), `fisher_` (the LDA head if active), `n_prototypes_`, `best_val_f1_`, `centroid_drift_`, `centroid_drift_per_proto_`, `_class_counts_`. Helpers: `summary()`, `predict_with_uncertainty()`.

---

## Design Decisions and What Didn't Work

### Worked

**Separate AVM and KNN feature spaces** — when both heads scored the same vector, fusion just averaged one geometry; splitting them gives the fusion real signal. The central architecture.

**Configurable heads/branches over hand-crafted interaction features** — the head/branch design lets a view contribute as its own late-fused signal. Augmenting the AVM vector with extra views directly *wrecks* it (covariance blow-up, transform mismatch, ‖x‖² pollution), which is the whole reason views go in the flat KNN space or as separate heads.

**Robust N-head fusion** — floor + shrink-to-uniform + margin. The unregularised selector overfit tiny imbalanced val splits and zeroed a head; the robust version fails safe and preserved the gains where heads genuinely help (Breast +0.028).

**Independent sigmoid gates, per-prototype RDA covariance, float64 Mahalanobis, `val_macro_bias`** — as before; all retained.

### Didn't Work / Retired

**Triangle interaction scoring** — real gains in its era, removed in Phase 9. The decline was over-determined: the triangle was a hand-crafted degree-2 interaction that **per-prototype Mahalanobis subsumes** (the precision matrix *is* the complete learned degree-2 interaction space), and triangles are intrinsically **ill-conditioned in exactly the imbalanced/multimodal regime** the project targets (near-collinear centroids → high-variance area/angle terms). The dual-boundary layout bug and the dead separation losses likely accelerated the visible failure, but the root cause is subsumption + regime-induced degeneracy, not a single broken fix.

**Fisher head on multimodal severe imbalance** — a global-linear discriminant is the wrong shape there; even with the robust fusion preventing collapse, it dilutes (Red Wine −0.06). Keep it for clean/moderate sets only.

**Static centroid gravity (`prior_temp>0`)** — strictly dominated on severe imbalance; `α=0` is the frontier peak in both directions. The fused posterior is too low-dynamic-range for a prior reweight to act only "at the margin"; it swamps the likelihood. Use `per_class_tau` instead.

**Earlier retirees** — kernel/RBF triangle, triplet (C(F,3) blow-up), bilinear-W, MLP λ gate, branch pruning.

---

## Open Items

| Item | Status | Notes |
|------|--------|-------|
| ArrivalType benchmark + baselines | Pending | Run `benchmark.py` and `baselines.py` with the CSV; the regime AVNN is built for and the real verdict |
| Re-sweep `intra_sep` | Pending | Now a live gradient; default 0.05 likely suboptimal for `n_prototypes=2` — use `sweep_intra_sep.py` (expect ~0 to win) |
| Confidence-gated gravity | Not recommended | The frontier sweep argues against it; minority lives in the uncertain region, so gated gravity still harms recall. Calibration + per-class thresholds is the principled successor |
| Learned low-rank metric `L` | Idea | Generalises diagonal feature weights to feature *combinations* (cross-feature engineering) without the per-class O(d²) variance cost; rows of `L` inspectable |
| Precision-matrix eigen read-out | Idea | Surface Σ⁻¹ eigenstructure as built-in cross-feature importance (no retraining) |
| Geodesic head validation | Pending | Heaviest, least-validated head; A/B where manifold structure is plausible |

---

## Files

| File | Purpose |
|------|---------|
| `LearningAVNN.py` | Model — import and use this |
| `benchmark.py` | Canonical per-dataset benchmark (one tuned config each) |
| `benchmark_heads.py` | Variant comparison: baseline vs +fisher / enriched-knn / both / +geodesic |
| `baselines.py` | Standard models (LogReg, LDA, GaussNB, KNN, SVM-RBF, RandForest, HistGBM), same datasets/CV |
| `sweep_prior_temp.py` | Centroid-gravity (`prior_temp`) × `per_class_tau` frontier sweep |
| `sweep_intra_sep.py` | Sweep the now-live `intra_sep` knob |
| `geometric_views.py` | Standalone Log/CLR/Rank/Fisher/Geodesic view transformers |
| `test_avnn.py` | Regression tests for the Phase-10 correctness fixes |
| `test_heads.py` | Heads/branches sanity + value CV |
| `test_views.py` | KNN-probe view screening |
| `requirements.txt` | `torch`, `numpy`, `scikit-learn`, `pandas`, `faiss-cpu` |
