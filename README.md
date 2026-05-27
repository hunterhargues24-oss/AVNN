# LearningAVNN

**A geometric tabular classifier with learnable distance metrics, interaction scoring, and imbalance handling.**

`sklearn`-compatible · pure PyTorch training · FAISS inference · 36 parameters · 1155 lines

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

LearningAVNN (Learning Angular Vector Neural Network) is a geometric distance-based tabular classifier. Instead of learning a standard decision boundary, it learns *what kind of distance* matters — transforming features into a multi-branch geometric space, then measuring how close a sample is to learned class prototypes.

The core question it answers: **does this sample's feature profile look like the profile of class k's prototype?** Both absolute position *and* relative feature relationships are measured.

It was designed specifically for imbalanced multi-class tabular classification, where tree ensembles and kernel methods struggle to recover rare class signal without sacrificing majority class accuracy.

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
    use_triangle=True, tri_weight=0.3,
    val_macro_bias=0.8,
    feat_weight_reg=0.01,
)

# Imbalanced dataset — severe (<2% minority)
model = LearningAVNN(
    ema_centroids=True,
    use_triangle=True, tri_weight=0.3,
    mahal_alpha=0.3,
    val_macro_bias=0.9,
    k=10,
    use_ivf=True,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)
print(model.summary(feature_names))
```

---

## Architecture

### Pipeline

```
Raw X  →  MinMax normalise [-1,1]  →  Six-branch combined vector (6F)
       →  AVM: inverse-distance to K×m learnable prototypes
       →  [Triangle scoring blend]
       →  KNN: FAISS nearest-neighbour vote
       →  λ·AVM + (1-λ)·KNN   (λ auto-tuned or entropy-gated)
       →  Mahalanobis correction at inference
```

### Six Branches

Each branch transforms normalised features differently. Branch weights are independent sigmoid gates (not softmax) normalised to sum to 1, preventing zero-sum competition. `boundary` and `tanh_arccos` are initialised with a positive bias — they consistently dominate on imbalanced datasets.

| # | Name | Formula | Signal |
|---|------|---------|--------|
| 0 | linear | `x_f · √fw` | Direction + magnitude |
| 1 | circular | `√(1-x²) · √fw` | Symmetric extremeness; peaks at boundaries |
| 2 | boundary | `√(‖x‖²-2x+1)` | Cross-feature coupling via global ‖x‖² |
| 3 | shape | `(x-μ)/σ` per sample | Within-sample relative profile |
| 4 | quadratic | `x² · √fw` | Nonlinear extremeness |
| 5 | tanh_arccos | `tanh(arccos(x)·0.8)·√fw` | Monotone nonlinear; amplifies centre differences |

**Per-feature weights** `fw` are learned alongside branch weights. The combined vector is `6F` wide.

### Learnable Prototypes

Centroids are `nn.Parameter` tensors of shape `(K, m, F)` where K=classes and m=prototypes per class. They are updated by backprop plus optional EMA stabilisation.

- `n_prototypes=1` — single centroid per class (default)
- `n_prototypes>1` — mixture of prototypes; models multimodal class distributions. Intra-class separation (`intra_sep`) pushes prototypes apart; soft EMA assignment pulls each toward its nearest sub-cluster.

### AVM Scoring

$$\text{score}_k(x) = \sum_j \frac{1}{d(x, \mu_{kj})\,/\,\tau_k + \varepsilon}$$

During training: Euclidean distance in the 6F combined space.  
At inference: **full RDA Mahalanobis** with α-blend of per-class QDA and pooled LDA:

$$\Sigma_k = (1-\alpha)\,S_k + \alpha\,S_{\text{pooled}}$$

Cholesky inversion in float64 prevents overflow on high-dimensional combined vectors.

### Triangle Scoring (`use_triangle=True`)

Pairwise deviation cross-product for all F(F-1)/2 feature pairs. Both sample and centroid are centred by their own feature means before scoring — this **eliminates false penalties** from absolute scale differences between samples of the same class.

For pair (a, b):

$$\text{cross}_{ab,k} = (x_a - \bar{x})(\mu_{kb} - \bar{\mu}_k) - (x_b - \bar{x})(\mu_{ka} - \bar{\mu}_k)$$

Zero when the sample's relative feature profile matches the centroid's for this pair. Each pair has a learnable weight (`log_w_pairs`) — the top pairs are directly inspectable and domain-validatable.

Blended with geometric AVM score: `(1-tri_weight)·AVM + tri_weight·triangle`.

**Proven impact: +0.058 macro F1 on moderate imbalance (4/82/14%), +0.047 on severe (60/25/10/5%).**

### Lambda Gate

Post-training AVM/KNN blend optimised on the validation set.

| Mode | Behaviour |
|------|-----------|
| `entropy_lambda=False` | Scalar λ grid-searched from `lam_floor` to 1.0 |
| `entropy_lambda=True` | Per-sample λ(x) based on AVM entropy H(x) |

Entropy gate: `λ(x) = lam_confident − (lam_confident − lam_uncertain) · H(x)`

High AVM entropy (uncertain, near decision boundary) → low λ → trust local KNN more.  
Low AVM entropy (confident, interior of cluster) → high λ → trust global AVM more.

### Training Losses

| Loss | Purpose | Key parameter |
|------|---------|---------------|
| Class-weighted NLL | Imbalance correction | `weight_cap` |
| Label smoothing | Prevent overconfidence | `label_smoothing` |
| Ordinal EMD | Penalise ordinal mistakes proportionally | `ordinal_weight` |
| Supervised Contrastive | Pull same-class embeddings together | `supcon_weight` |
| Gram matrix ortho | Decorrelate six branches | `ortho_reg` |
| Centroid anchor | Keep prototypes near class means | `centroid_reg` |
| Centroid separation | Push class prototypes apart | `centroid_sep` |
| Intra separation | Push within-class prototypes apart (m>1) | `intra_sep` |
| Feature weight reg | Prevent feature weight collapse | `feat_weight_reg` |

### Inference

```python
predict_proba(X):
    Xn       = normalise(X)                    # train statistics
    combined = _make_combined(Xn)              # numpy, 6F
    avm      = Mahalanobis inverse-distance    # to K×m prototypes
    knn      = FAISS KNN vote                  # IVF or FlatL2
    λ        = entropy_gate(avm) or scalar
    return λ·avm + (1-λ)·knn
```

---

## Evolution History

### Phase 1 — Foundation

Started as a three-branch static classifier (`FastTriBranchAVNN`) with fixed branch weights: angular (arccos), euclidean (tanh_arccos), and shape (z-score). No gradient, no training — just class mean centroids and FAISS KNN. Competitive with SVM on balanced data; weak on imbalance.

**Key finding:** FastTriBranchAVNN outperformed the early learnable version on Wine cultivar (0.9884 vs 0.9831) because the shape (z-score) branch captured within-sample relative profiles that the learnable model's arccos-based branches couldn't. This drove the shape branch into the learnable model permanently.

### Phase 2 — Learnable Branches

Introduced gradient-based branch weight learning, per-feature weights, and learnable centroids. Upgraded softmax branch competition to **independent normalised sigmoid gates** — this eliminated zero-sum competition where increasing one branch necessarily suppressed others.

Branch weight initialisation: `boundary` and `tanh_arccos` biased to sigmoid(0.5)≈0.622 vs sigmoid(0)=0.5 for others, based on observed dominance on imbalanced datasets.

**Ablation finding:** Reducing from 6 to 4 branches (dropping linear and quadratic) degraded performance despite those branches having near-uniform weights. Even redundant branches enriched the KNN geometry — the FAISS index benefits from higher-dimensional combined spaces even when the gradient cannot differentiate the branches.

### Phase 3 — Distance Upgrade

Replaced diagonal covariance with **full RDA Mahalanobis** (α-blend of per-class QDA and pooled LDA). Cholesky inversion for numerical stability. Float64 intermediates required for high-dimensional combined vectors (discovered via overflow errors on digits dataset, 384-dim combined space).

### Phase 4 — Imbalance Toolkit

Added:
- **Supervised Contrastive Loss (SupCon)** — pulls same-class embeddings together in the 6F combined space. Effective when ≥2 rare samples appear per batch; effectively a no-op for <0.4% minority at batch_size=256.
- **EMA centroid stabilisation** — exponential moving average of embedded training samples, preventing minority centroids from drifting under sparse gradient updates.
- **Ordinal EMD loss** — Earth Mover's Distance for ordered class labels (e.g. wine quality), penalising misclassification proportionally to class distance.
- **Per-class temperature** (`per_class_tau`) — minority classes learn softer scoring, preventing them from being crushed by majority class confidence.

### Phase 5 — Mixture of Prototypes

Extended centroids from `(K, F)` to `(K, m, F)` — each class can have m learnable prototypes. Soft EMA assignment updates each prototype toward its nearest sub-cluster. The medium-quality wine class (82% of red wine data) spans q=5 and q=6 wines that cluster differently; a single centroid averaged them into an incoherent mean. Multi-prototype addresses this.

### Phase 6 — Triangle Interaction Scoring

**Key architectural advance.** Pairwise cross-products between sample and centroid feature deviations. Initially implemented as raw ratio comparison (`x_a·μ_kb - x_b·μ_ka`), then upgraded to **deviation space** (`(x_a-x̄)·(μ_kb-μ̄_k) - (x_b-x̄)·(μ_ka-μ̄_k)`) which eliminates false penalties from absolute scale differences between samples of the same class.

The deviation version improved results on severe imbalance over the raw version: synth-severe at tri=0.5 went from -0.005 to +0.013 — a complete sign reversal.

### Phase 7 — Lambda Gate

Added entropy-based per-sample lambda gate. AVM entropy H(x) ∈ [0,1] measures how uncertain the AVM is — near decision boundaries entropy is high, meaning local KNN structure is more reliable than the global centroid geometry. The gate smoothly transitions between `lam_confident` (interior points) and `lam_uncertain` (boundary points).

### Phase 8 — Cleanup

**Removed:**
- Kernel trick (RBF replacement for triangle) — loses interpretability; triangle without kernel is cleaner and competitive
- Triplet scoring (3D cross-product for feature triplets) — C(F,3) combinatorial explosion; gradient spreads too thin (165 triplets for F=11, 1140 for F=20); no meaningful improvement
- Bilinear W matrix scoring — zero-gradient initialisation bug exposed fundamental learning difficulty; even after fixing, combining with triangle hurt rather than helped due to gradient competition in shared deviation space
- Branch pruning — never improved results in testing; added ~80 lines of training loop complexity
- MLP lambda gate — no improvement over entropy gate; added parameters with no payoff
- `nlist`, `nprobe`, `auto_lambda`, `lam_search_step`, `val_every`, `lam_init` — hardcoded sensible defaults
- `val_acc_weight` + `val_macro_weight` + `val_weighted_weight` → consolidated to single `val_macro_bias` float

---

## Benchmarks

All results are 5-fold cross-validation macro F1 unless noted. Baselines use sklearn defaults with StandardScaler where required.

### Balanced Datasets

| Model | Iris | Wine (cultivar) | Digits |
|-------|------|-----------------|--------|
| LearningAVNN | **0.9598** | **0.9831** | **0.9844** |
| SVM-RBF | 0.9599 | 0.9834 | 0.9839 |
| Random Forest | 0.9464 | 0.9784 | 0.9788 |
| GBM | 0.9463 | 0.9515 | 0.9649 |
| XGBoost | 0.9456 | 0.9608 | 0.9621 |
| Logistic Reg | 0.9532 | 0.9829 | 0.9710 |

LearningAVNN ties SVM on balanced data. GBM notably underperforms on Wine cultivar (0.9515 vs 0.9831) — the geometric distance approach handles the 3-class cultivar separation well.

### Imbalanced Datasets (Synthetic)

| Model | Synth-moderate (4/82/14%) | Synth-severe (60/25/10/5%) |
|-------|--------------------------|---------------------------|
| LearningAVNN (tri=0.3) | **0.6279** | **0.7574** |
| XGBoost | 0.6755 | 0.7787 |
| SVM-RBF | 0.5831 | **0.8039** |
| Random Forest | 0.6202 | 0.7401 |
| GBM | 0.6366 | 0.7441 |
| Logistic Reg | 0.5046 | 0.6052 |

Synth-moderate mirrors red wine (4% minority). Synth-severe mirrors ArrivalType (5% rarest class).

SVM-RBF leads on severe imbalance via implicit kernel interactions. XGBoost leads on moderate via threshold-based tree splits. LearningAVNN with triangle scoring is competitive with tree ensembles and significantly ahead of linear and kernel-free baselines.

### Triangle Scoring Impact (deviation cross-product)

| Dataset | Baseline | +Triangle (tri=0.3) | Δ |
|---------|----------|---------------------|---|
| Synth-moderate | 0.5700 | **0.6279** | +0.058 |
| Synth-severe | 0.7108 | **0.7574** | +0.047 |
| Iris | 0.9598 | 0.9598 | 0.000 |
| Wine cultivar | 0.9831 | 0.9831 | 0.000 |
| Digits | 0.9844 | 0.9821 | −0.002 |

Triangle scoring helps imbalanced datasets where class separation involves relative feature ratios. It is neutral on well-separated balanced data and slightly negative on high-dimensional balanced data (64 features → 2016 pairs, gradient too diluted).

### Key Architectural Comparisons

| Change | Impact |
|--------|--------|
| Full RDA Mahalanobis vs diagonal | +0.02–0.03 macro F1 on imbalanced |
| Deviation triangle vs raw triangle | Synth-severe tri=0.5: −0.005 → +0.013 |
| 6 branches vs 4 branches | 6 consistently better despite redundancy |
| Independent sigmoid gates vs softmax | Allows boundary + tanh_arccos to both be strong |
| Shape branch (z-score) vs no shape | Wine cultivar: 0.9884 → 0.9831 without it |

---

## Parameter Reference

### Core Training

| Parameter | Default | Notes |
|-----------|---------|-------|
| `k` | 5 | KNN neighbours |
| `lr` | 1e-3 | Adam learning rate |
| `epochs` | 300 | Max training epochs |
| `batch_size` | 256 | Mini-batch size |
| `patience` | 60 | Early stopping patience (val checks) |
| `val_fraction` | 0.15 | Held-out fraction for early stopping + λ search |
| `random_state` | None | Reproducibility seed |
| `verbose` | False | Print training progress |

### Regularisation

| Parameter | Default | Notes |
|-----------|---------|-------|
| `label_smoothing` | 0.05 | NLL label smoothing |
| `weight_cap` | 1.5 | Max class weight (prevents extreme minority over-weighting) |
| `weight_decay` | 1e-4 | Adam L2 regularisation |
| `feat_weight_reg` | 0.05 | Per-feature weight regularisation; use 0.01 for imbalanced |
| `ortho_reg` | 0.01 | Gram matrix branch decorrelation penalty |

### Prototypes

| Parameter | Default | Notes |
|-----------|---------|-------|
| `centroid_reg` | 0.1 | Anchors prototypes near class means |
| `centroid_sep` | 0.05 | Inter-class prototype separation |
| `n_prototypes` | 1 | Prototypes per class; >1 for multimodal classes |
| `intra_sep` | 0.05 | Intra-class prototype spread (only active when n_prototypes>1) |

### Interaction Scoring

| Parameter | Default | Notes |
|-----------|---------|-------|
| `use_triangle` | False | Enable pairwise deviation cross-product scoring |
| `tri_weight` | 0.5 | Triangle blend: 0=pure AVM, 1=pure triangle. Use 0.3 for imbalanced. |

### Distance Metric

| Parameter | Default | Notes |
|-----------|---------|-------|
| `mahalanobis` | True | Full RDA Mahalanobis at inference |
| `mahal_reg` | 1e-6 | Ridge regularisation on covariance |
| `mahal_alpha` | 0.5 | QDA↔LDA blend; 0=per-class, 1=pooled. Use 0.2–0.3 for severe imbalance. |

### Lambda Gate

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lam_floor` | 0.5 | Minimum AVM weight in scalar λ search |
| `entropy_lambda` | False | Enable per-sample entropy-gated λ |
| `lam_confident` | 0.90 | λ when AVM entropy is low (confident) |
| `lam_uncertain` | 0.40 | λ when AVM entropy is high (uncertain) |

### Imbalance Handling

| Parameter | Default | Notes |
|-----------|---------|-------|
| `ordinal` | False | Ordinal EMD loss for ordered class labels |
| `ordinal_weight` | 0.5 | EMD loss blend weight |
| `supcon` | False | Supervised contrastive loss |
| `supcon_weight` | 0.3 | SupCon loss blend weight |
| `supcon_temp` | 0.1 | SupCon temperature |
| `ema_centroids` | False | EMA centroid stabilisation |
| `ema_beta` | 0.9 | EMA decay rate |
| `per_class_tau` | True | Per-class temperature scaling |
| `val_macro_bias` | 0.5 | Val scoring: 0=accuracy, 1=macro F1 |

### FAISS

| Parameter | Default | Notes |
|-----------|---------|-------|
| `use_ivf` | True | IVF approximate index; auto-falls back to FlatL2 for small datasets |

---

## Design Decisions and What Didn't Work

### Worked

**Independent sigmoid gates over softmax** — Softmax creates zero-sum competition. When boundary AND tanh_arccos both needed to be high simultaneously (as on imbalanced wine datasets), softmax forced them to compete. Normalised sigmoid resolved this.

**Deviation space for triangle scoring** — Raw ratio comparison (`x_a·μ_b - x_b·μ_a`) penalises samples that match the centroid's profile at a different absolute scale. Deviation-space centring (`(x_a-x̄)·(μ_b-μ̄)`) eliminates this false penalty. The change converted synth-severe tri=0.5 from −0.005 to +0.013.

**Keeping all 6 branches despite redundancy** — Ablation showed removing linear and quadratic (near-uniform weights on balanced data) degraded holdout performance. Partially redundant branches still enrich the KNN geometry by increasing combined vector dimensionality even when the gradient cannot differentiate them.

**Float64 Mahalanobis intermediates** — Float32 overflows on high-dimensional combined vectors (384-dim for digits). Overflow → NaN → silent divide-by-zero in scoring. Promoting to float64 for the matrix multiply fixed this.

**val_macro_bias over three separate weights** — Three weights (`val_acc_weight`, `val_macro_weight`, `val_weighted_weight`) created a 3D search space with correlated parameters. Single `val_macro_bias` float covers the same intent with 1 degree of freedom.

### Didn't Work

**MLP lambda gate** — A small network (D→16→1) trained on the val set to predict per-sample λ. Added ~100 parameters and 100 training epochs. No meaningful improvement over the zero-parameter entropy gate. Removed.

**Kernel trick (RBF replacement for triangle)** — Replacing inverse-distance scoring with `exp(-d²/2σ²)` with learnable σ per class. Improved synth-severe by +0.025 but lost all triangle pair interpretability — the W matrix and top pairs are among the most actionable outputs the model produces. Not worth the tradeoff.

**Triplet scoring (3D cross-product)** — For F=11: 165 triplets vs 55 pairs. For F=20: 1140 triplets. The gradient spreads too thin across 165 learnable weights to find meaningful signal within 300 epochs. No improvement on any dataset; significant computation overhead.

**Bilinear W matrix** — F×F asymmetric interaction matrix. Two problems: (1) `W²` initialisation means zero gradient at `W=0`, requiring random init to break symmetry; (2) operates in the same deviation space as triangle, creating gradient competition through shared centroids. Combining with triangle hurt rather than helped on synth-moderate (0.5771 vs 0.6279 for triangle alone). Removed.

**Branch pruning** — Freezing low-weight branches after warmup. Never improved results across any dataset; added ~80 lines of training loop complexity. The partially-redundant branches contribute to KNN geometry even at low gradient weight.

**Softmax branch competition (early)** — Original softmax gate meant boundary and tanh_arccos competed directly. Replaced with normalised sigmoid.

---

## Open Items

| Item | Status | Notes |
|------|--------|-------|
| `n_prototypes > 1` | Implemented, untested | Multimodal class distributions; red wine q=5 vs q=6 sub-clusters are the target use case |
| Separate KNN feature space | Designed, not built | AVM and KNN currently use identical combined vectors — no genuine disagreement possible. Separate representation for KNN could let it correct AVM errors on minority classes |
| Wine / ArrivalType with triangle | Pending | Need to run `wine_arrival_test.py` on the full datasets |
| German Credit / Pima Diabetes | Pending network | Standard imbalanced benchmarks via `fetch_openml`; blocked by sandbox network |

---

## Files

| File | Purpose |
|------|---------|
| `LearningAVNN.py` | Model — import and use this |
| `auto_lambda_test.py` | Full benchmark: Iris, Wine, Breast Cancer, Red Wine, White Wine, ArrivalType |
| `quick_test.py` | Fast benchmark: Iris, Wine, Digits, Synth-moderate, Synth-severe |
| `wine_arrival_test.py` | Red Wine, White Wine, ArrivalType with triangle scoring |
| `benchmark_all_models.py` | LearningAVNN vs XGBoost, LightGBM, SVM, RF, GBM |
| `requirements.txt` | `torch`, `numpy`, `scikit-learn`, `pandas`, `faiss-cpu` |
| `ARCHITECTURE.md` | Detailed architecture reference |
