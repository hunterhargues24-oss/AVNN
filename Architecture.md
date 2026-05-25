# AutoLambdaAVNN — Architecture Reference

> A geometric classifier that learns angular distance metrics over class centroids, blended with FAISS-accelerated nearest neighbours. Designed for tabular classification with built-in handling for imbalanced data.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Step 1 — Normalisation](#2-step-1--normalisation)
3. [Step 2 — The Four Branches](#3-step-2--the-four-branches)
4. [Step 3 — The Combined Vector](#4-step-3--the-combined-vector)
5. [Step 4 — What Gets Learned](#5-step-4--what-gets-learned)
6. [Step 5 — Training Loop](#6-step-5--training-loop)
7. [Step 6 — Centroid Corrections](#7-step-6--centroid-corrections)
8. [Step 7 — Inference](#8-step-7--inference)
9. [Step 8 — Loss Function](#9-step-8--loss-function)
10. [Component Reference](#10-component-reference)
11. [Parameter Reference](#11-parameter-reference)
12. [Design Decisions and Tradeoffs](#12-design-decisions-and-tradeoffs)

---

## 1. Overview

AutoLambdaAVNN is a hybrid of two distance-based classifiers:

- **AVM (Angular Vector Machine)** — votes by inverse distance to K class centroids
- **KNN** — votes by inverse distance to the k nearest training points

Both use the same learned distance metric, so they are always geometrically consistent. The blend between them (`lambda`) is tuned automatically on a validation set after training.

```
Input X
   │
   ▼
Normalise → [-1, 1]
   │
   ▼
Four-Branch Combined Vector  (learned geometry)
   │
   ├──► AVM: distance to K centroids  ──┐
   │                                    ├──► λ * AVM + (1-λ) * KNN → predict
   └──► KNN: distance to k neighbours ──┘
```

The model is trained on the AVM branch only (fast — O(batch × K)), then the KNN branch is added at inference via a FAISS index built from the same learned geometry.

---

## 2. Step 1 — Normalisation

All features are normalised to **[-1, 1]** using per-feature min-max scaling:

$$x_f^{norm} = -1 + 2 \cdot \frac{x_f - \min_f}{\max_f - \min_f}$$

**Why [-1, 1] and not [0, 1]?**

The arccos function used in the angular branches is defined on [-1, 1] and gives angles in [0, π]. This full range captures sign (positive vs negative deviation from the feature midpoint) as well as magnitude. In [0, 1] space arccos only covers [0, π/2], losing the sign information.

The normalisation is fit on the training split only and applied to validation and test sets — no data leakage.

---

## 3. Step 2 — The Four Branches

Each normalised feature vector is described by four geometric signals. Each branch captures a fundamentally different aspect of where a point sits in feature space.

### Branch 1 — Direction (`phi_dir`)

$$\phi_{dir,f}(x) = \arccos(x_f) \in [0, \pi]$$

**What it captures:** Direction-aware position along the feature axis.
- $x_f = +1$ → $\phi_{dir} = 0$ (at positive boundary)
- $x_f = 0$ → $\phi_{dir} = \pi/2$ (at centre)
- $x_f = -1$ → $\phi_{dir} = \pi$ (at negative boundary)

Points at +0.8 and -0.8 look completely different here (0.64 vs 2.50 radians). This branch knows both *how extreme* a feature is and *which direction*.

### Branch 2 — Magnitude (`phi_mag`)

$$\phi_{mag,f}(x) = \arccos(|x_f|) \in [0, \pi/2]$$

**What it captures:** Extremeness regardless of direction.
- $x_f = \pm 1$ → $\phi_{mag} = 0$ (at either boundary)
- $x_f = 0$ → $\phi_{mag} = \pi/2$ (at centre, farthest from any boundary)

Points at +0.8 and -0.8 look *identical* here — both are 0.64 radians from their nearest boundary. This branch is direction-blind. It provides orthogonal information to `phi_dir` — together they uniquely encode both sign and extremeness without redundancy.

**Why these two are non-redundant:**

$\arccos(-x_f) = \pi - \arccos(x_f)$, so the "negative corner angle" would be perfectly correlated with `phi_dir`. But $\arccos(|x_f|)$ is not a linear function of $\arccos(x_f)$ — it explicitly captures the distance to the *nearest* boundary, which is a genuinely new signal.

### Branch 3 — Euclidean (`euc`)

$$euc_f(x) = \tanh(\arccos(x_f) \cdot 0.8) \in [0, \tanh(0.8\pi)]$$

**What it captures:** A nonlinear compressed distance from the positive boundary.

The `tanh` compression creates a different curvature from `phi_dir`. Near $x_f = +1$, `euc` is nearly linear. Near $x_f = -1$, `tanh` saturates, compressing the range. This means the metric "notices" small differences near +1 more than large differences near -1 — a different sensitivity profile from the raw arccos.

**Implementation note:** `phi_dir` and `euc` both require $\arccos(x_f)$. This is computed once and reused — one arccos call instead of two.

### Branch 4 — Shape

$$shape_f(x) = \frac{x_f - \bar{x}}{std(x)}$$

**What it captures:** The within-sample relative feature profile. This is the per-sample z-score — it strips out the absolute magnitude of all features and captures only their relative pattern.

A wine with alcohol=14, acidity=0.8 and a wine with alcohol=7, acidity=0.4 have the same shape (alcohol twice acidity) but completely different `phi_dir` values.

**Why no feature weights on shape:** Shape is already mean-centred and variance-normalised within each sample. Applying per-feature weights would partly undo this normalisation.

---

## 4. Step 3 — The Combined Vector

The four branches are concatenated into a single vector with learned scaling:

$$v(x) = \left[\sqrt{w_d} \cdot \sqrt{fw} \cdot \phi_{dir},\;\; \sqrt{w_m} \cdot \sqrt{fw} \cdot \phi_{mag},\;\; \sqrt{w_e} \cdot \sqrt{fw} \cdot euc,\;\; \sqrt{w_s} \cdot shape\right]$$

Where:
- $w_d, w_m, w_e, w_s$ are branch weights (sum to 1, learned via softmax)
- $fw_f$ are per-feature weights (positive, learned via softmax × F so uniform = 1.0 each)
- $\sqrt{\cdot}$ scaling ensures squared L2 in this space gives branch-weighted distance

**Why sqrt scaling?**

The squared L2 distance between two combined vectors is:

$$\|v(x) - v(y)\|^2 = w_d \sum_f fw_f (\phi_{dir,f}(x) - \phi_{dir,f}(y))^2 + w_m \sum_f fw_f (\phi_{mag,f}(x) - \phi_{mag,f}(y))^2 + \ldots$$

This is exactly the branch-weighted, feature-weighted sum of squared per-branch distances. The sqrt scales encode the weighting in the vector itself — so FAISS (which computes L2 distance) automatically applies the correct metric.

**Critical property:** This combined vector is built identically in PyTorch (training) and NumPy (inference). There is no train/inference metric mismatch.

---

## 5. Step 4 — What Gets Learned

The model learns five groups of parameters via gradient descent:

| Parameter | Shape | Initialisation | What it controls |
|-----------|-------|----------------|-----------------|
| `raw_w_d/m/e/s` | scalar × 4 | 0.0 | Branch weights via softmax |
| `raw_feat_w` | (F,) | zeros | Per-feature weights via softmax × F |
| `log_tau` | scalar or (K,) | -0.5 | AVM scoring temperature |

**Branch weights** — softmax ensures they sum to 1:

$$w = \text{softmax}([raw\_w_d, raw\_w_m, raw\_w_e, raw\_w_s])$$

Starting at 0.0 means all branches are equal at initialisation (softmax of zeros = 0.25 each).

**Feature weights** — softmax then scaled so uniform = 1.0:

$$fw = \text{softmax}(raw\_feat\_w) \times F$$

This means a feature weight of 2.0 contributes twice as much to the distance as a feature at the uniform baseline of 1.0.

**Temperature (tau)** — controls AVM scoring sharpness:

$$\tau = \exp(log\_\tau)$$

Small tau: distances are divided by a small number, making the inverse-distance scores very peaked — the model is confident. Large tau: softer, more uniform scoring. With `per_class_tau=True`, each class gets its own tau — minority class centroids (noisy, few samples) can learn higher tau (softer) while majority class centroids learn lower tau (sharper).

---

## 6. Step 5 — Training Loop

Training is pure AVM — no KNN in the forward pass. This keeps training at O(batch × K × F) rather than O(batch × N × F).

### Forward pass

For a mini-batch of normalised training points:

$$d_{ij} = \|v(x_i) - v(\mu_j)\|_2 \quad \text{(distance from point i to centroid j)}$$

$$p_{ij} = \frac{1/({d_{ij}/\tau_j + \epsilon})}{\sum_k 1/({d_{ik}/\tau_k + \epsilon})} \quad \text{(softmax over inverse distances)}$$

Where $\mu_j$ is the class mean in normalised space (fixed buffer, not learned).

### Centroid cache

Computing $v(\mu_j)$ for all K centroids is cheap (K is small) but still unnecessary to repeat for every mini-batch within an epoch since the weights don't change between batches. The centroid features are cached at the start of each epoch and invalidated after each optimizer step.

### Early stopping

Validation F1 is computed using the composite metric:

$$score = w_{acc} \cdot acc + w_{macro} \cdot F1_{macro} + w_{weighted} \cdot F1_{weighted}$$

Default weights are equal (1/3 each). On imbalanced datasets, `suggest_kwargs()` adjusts these — for moderate imbalance, macro F1 gets 0.50 weight to prioritise minority class detection without ignoring accuracy entirely.

A WARMUP of 20 epochs runs before patience counting begins, preventing early stopping from firing before the model has had time to converge.

### Cosine annealing scheduler

Learning rate follows cosine annealing with warm restarts (`T_0=30, T_mult=2`). The learning rate cycles from `lr` down to `1e-5` and back, with each cycle twice as long as the previous. This helps escape local optima in the branch weight landscape.

---

## 7. Step 6 — Centroid Corrections

After training, two post-hoc corrections are applied to the centroid positions in combined space before building the FAISS index. Both operate on the geometry the model actually learned.

### James-Stein Shrinkage

The raw class mean is an unreliable estimate when $n_k$ is small. James-Stein shrinkage blends each centroid toward the global data mean:

$$\hat{\mu}_k = (1 - s_k) \cdot \mu_k^{raw} + s_k \cdot \bar{\mu}^{global}$$

$$s_k = \frac{1}{1 + n_k / \text{shrinkage\_strength}}$$

**Effect by class size:**

| $n_k$ | $s_k$ (strength=20) | Behaviour |
|-------|---------------------|-----------|
| 10,000 (majority) | 0.002 | Centroid barely moves |
| 50 (moderate minority) | 0.286 | Centroid pulled 29% toward global mean |
| 10 (rare minority) | 0.667 | Centroid pulled 67% toward global mean |

Minority centroids move toward where the overall data lives — expanding their geometric catchment area. `shrinkage_strength=0` disables this entirely.

### Diagonal Mahalanobis Distance (`mahalanobis=True`)

Standard Euclidean distance treats all combined-vector dimensions equally. Mahalanobis distance scales each dimension by the per-class variance, making the metric adapt to the actual shape of each class cluster:

$$d_{Mahal}(x, \mu_k) = \sqrt{\sum_d \frac{(x_d - \mu_{k,d})^2}{\sigma_{k,d}^2}}$$

Where $\sigma_{k,d}^2$ is the variance of class k's training samples along dimension d (in combined space), regularised toward the global variance:

$$\sigma_{k,d}^2 = (1 - \alpha) \cdot \hat{\sigma}_{k,d}^2 + \alpha \cdot \sigma_{global,d}^2$$

**Effect:** If class k has high variance along a direction (the class is spread out there), distances in that direction are down-weighted. If class k is tightly clustered along a direction, distances there are up-weighted. The metric adapts to the actual shape of each class cluster.

**Why diagonal only (not full covariance):** A full covariance matrix for K=4F=44 dimensions requires estimating $44 \times 44 = 1{,}936$ parameters per class. With 64 minority class samples this is badly underdetermined. Diagonal covariance requires only 44 parameters per class — always estimable.

---

## 8. Step 7 — Inference

### AVM scoring

$$p_{AVM}(k|x) \propto \frac{1}{d(v(x), \hat{\mu}_k) + \epsilon}$$

Where $\hat{\mu}_k$ is the shrinkage-corrected centroid and $d$ is either Euclidean or diagonal Mahalanobis depending on the `mahalanobis` flag.

### KNN scoring

The FAISS index is built from all training combined vectors $v(x_i)$. For a query point $x$:

1. Find k nearest training points by L2 distance in combined space
2. Weight each neighbour by inverse distance: $w_i = 1 / (d_i + \epsilon)$
3. Accumulate normalised weights into class probability bins

$$p_{KNN}(k|x) = \sum_{i \in N_k(x)} \frac{w_i}{\sum_j w_j}$$

Where $N_k(x)$ is the set of neighbours belonging to class k.

FAISS IVF (Inverted File Index) is used for large datasets. Rather than computing all N distances, it clusters training points into `nlist` cells and only searches `nprobe` cells near the query point. Inference scales as O(log N) rather than O(N).

For very large test sets, FAISS search is batched in chunks of 50,000 to prevent the `std::bad_alloc` that occurs when allocating the full result matrix at once.

### Lambda blend

$$p(k|x) = \lambda \cdot p_{AVM}(k|x) + (1-\lambda) \cdot p_{KNN}(k|x)$$

$\lambda$ is found by a post-training grid search over [0.3, 1.0] with step 0.05, evaluated on the validation set using the composite metric. The search starts at 0.3 — pure KNN ($\lambda=0$) discards the learned AVM geometry entirely.

### FAISS IVF vs Flat index

| Dataset size | Index type | Behaviour |
|-------------|------------|-----------|
| < ~3,000 | `IndexFlatL2` | Exact search, always fast enough |
| ≥ ~3,000 | `IndexIVFFlat` | Approximate, O(nprobe/nlist × N) |

IVF requires training the quantizer on at least 39 × nlist samples. If the training set is too small, falls back to Flat automatically.

---

## 9. Step 8 — Loss Function

The training loss combines three terms:

$$\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{reg} + \mathcal{L}_{ordinal}$$

### Cross-entropy with label smoothing

$$\mathcal{L}_{CE} = (1 - \alpha + \alpha/K) \cdot \mathcal{L}_{NLL} - (\alpha/K) \cdot \mathbb{E}[\log p]$$

Where $\alpha$ is `label_smoothing` (default 0.05). This prevents the model from becoming overconfident on training labels. Class weights from sklearn's `balanced` strategy are applied to the NLL, capping at `weight_cap` to prevent extreme upweighting of tiny minority classes.

### Feature weight regularisation

$$\mathcal{L}_{reg} = \lambda_{reg} \cdot \frac{1}{F}\sum_f (fw_f - 1)^2$$

Penalises deviation of feature weights from the uniform baseline of 1.0. The gradient at $fw_f = 1.0$ is zero — no force to move from uniform. Features only diverge from 1.0 when the classification gradient justifies it. This prevents the overfit pattern of two features exploding to 4.5× and everything else collapsing.

### Ordinal loss — Earth Mover's Distance (`ordinal=True`)

For K ordered classes, standard NLL penalises all wrong predictions equally. Predicting class 2 when the truth is class 1 (adjacent) costs the same as predicting class 3 (far away). The EMD loss penalises in proportion to ordinal distance:

$$\mathcal{L}_{EMD} = \sum_{k=0}^{K-2} |CDF_p[k] - CDF_y[k]|$$

Where $CDF_p[k] = \sum_{j=0}^{k} p_j$ is the cumulative predicted probability and $CDF_y[k]$ is the step function at the true class.

**Example for K=3:**

| True class | Prediction | NLL cost | EMD cost |
|-----------|------------|----------|----------|
| 1 (medium) | [0.0, 0.9, 0.1] | low | low |
| 1 (medium) | [0.8, 0.1, 0.1] | high | 0.8 (1 step wrong) |
| 1 (medium) | [0.1, 0.1, 0.8] | high | 0.8 (1 step wrong) |

Both end-class mistakes cost the same under EMD. A model predicting [0.4, 0.1, 0.5] would have EMD = |0.4 - 0| + |0.9 - 1| = 0.5, penalising the high-quality prediction proportionally.

The final loss blends NLL and EMD:

$$\mathcal{L} = (1 - w_{ord}) \cdot \mathcal{L}_{CE} + w_{ord} \cdot \mathcal{L}_{EMD}$$

Only active when K > 2 — binary classification has no ordinal distance.

---

## 10. Component Reference

### How components interact

```
Training data
     │
     ▼
┌─────────────────────────────────────────────┐
│  NORMALISATION  [-1, 1] min-max per feature │
└─────────────────────────┬───────────────────┘
                          │
              ┌───────────▼───────────┐
              │   FOUR BRANCHES       │
              │  phi_dir  arccos(x)   │
              │  phi_mag  arccos(|x|) │
              │  euc      tanh_ac(x)  │
              │  shape    z-score(x)  │
              └───────────┬───────────┘
                          │
              ┌───────────▼────────────────┐
              │  COMBINED VECTOR           │
              │  [√w_d·√fw·φ_dir,          │
              │   √w_m·√fw·φ_mag,          │
              │   √w_e·√fw·euc,            │
              │   √w_s·shape]              │
              │                            │
              │  Learned: w_d,w_m,w_e,w_s  │
              │           fw_f  tau         │
              └──────┬────────────┬────────┘
                     │            │
          ┌──────────▼──┐   ┌─────▼───────────┐
          │  TRAINING   │   │  POST-TRAINING  │
          │  AVM only   │   │  Shrinkage      │
          │  NLL + EMD  │   │  Mahalanobis    │
          │  + reg loss │   │  Lambda search  │
          └─────────────┘   └─────┬───────────┘
                                  │
                       ┌──────────▼──────────┐
                       │    INFERENCE        │
                       │  λ·AVM + (1-λ)·KNN │
                       └─────────────────────┘
```

### When each correction fires

| Correction | When active | What it fixes |
|-----------|-------------|---------------|
| Class-weighted NLL | Always | Training signal for minority classes |
| Feature weight reg | Always | Prevents 2-feature overfit on small data |
| Label smoothing | Always | Overconfidence on training labels |
| Per-class tau | `per_class_tau=True` | Noisy minority centroids get softer scoring |
| Ordinal loss | `ordinal=True, K>2` | Adjacent class confusion penalised less |
| Shrinkage | `shrinkage_strength > 0` | Noisy minority centroids moved toward data centre |
| Mahalanobis | `mahalanobis=True` | Distance metric adapts to per-class cluster shape |
| Auto lambda | `auto_lambda=True` | AVM/KNN blend tuned per dataset |

---

## 11. Parameter Reference

### Core training

| Parameter | Default | Effect |
|-----------|---------|--------|
| `k` | 5 | KNN neighbours |
| `lr` | 1e-3 | Adam learning rate |
| `epochs` | 300 | Max training epochs |
| `batch_size` | 256 | Mini-batch size — larger is faster per epoch |
| `patience` | 60 | Early stopping epochs without improvement |
| `val_fraction` | 0.15 | Validation holdout — stratified by class |

### Regularisation

| Parameter | Default | Effect |
|-----------|---------|--------|
| `weight_cap` | 1.5 | Max class weight in NLL (prevents extreme minority upweighting) |
| `weight_decay` | 1e-4 | Adam L2 parameter regularisation |
| `feat_weight_reg` | 0.05 | L2 penalty on feature weights toward uniform — higher = less divergence |
| `label_smoothing` | 0.05 | Smoothing on NLL targets |

### Imbalance handling

| Parameter | Default | Effect |
|-----------|---------|--------|
| `shrinkage_strength` | 20.0 | Controls centroid pull toward global mean — 0 disables |
| `mahalanobis` | False | Enable diagonal Mahalanobis distance |
| `mahal_reg` | 1e-6 | Regularisation toward global variance — raise for small minority classes |
| `ordinal` | False | Enable EMD ordinal loss |
| `ordinal_weight` | 0.5 | Blend weight between NLL and EMD |
| `per_class_tau` | False | Per-class AVM temperature |

### Inference

| Parameter | Default | Effect |
|-----------|---------|--------|
| `lam_init` | 0.7 | Starting lambda — used directly if `auto_lambda=False` |
| `auto_lambda` | True | Grid search for best AVM/KNN blend on validation set |
| `lam_search_step` | 0.05 | Lambda grid resolution (0.05 = 15 candidates from 0.3 to 1.0) |
| `use_ivf` | True | FAISS IVF approximate index (fast for large datasets) |
| `nlist` | 1000 | IVF clusters — capped at n_train//39 |
| `nprobe` | 10 | IVF cells searched per query — higher = more accurate, slower |

### Validation metric weights

| Parameter | Default | Raise when |
|-----------|---------|------------|
| `val_acc_weight` | 0.33 | Severe imbalance (prevent macro F1 gaming) |
| `val_macro_weight` | 0.34 | Moderate imbalance (prioritise minority detection) |
| `val_weighted_weight` | 0.33 | Balanced data (equal weight to all three) |

---

## 12. Design Decisions and Tradeoffs

### Why pure AVM during training?

Training with KNN would require computing distances to all N training points per batch — O(batch × N × F). With N=200,000 and F=44 combined dimensions, this is impractical. Pure AVM training runs O(batch × K × F) where K is the number of classes (typically 2-10). After training, the FAISS index is built once and used at inference for O(log N) KNN lookup.

The tradeoff: the learned weights are optimised for AVM scoring, then applied to KNN at inference. Lambda search on the validation set finds the best blend of these two signals given the actual learned geometry.

### Why arccos rather than cosine similarity?

Cosine similarity computes a single global angle between the full feature vector and a reference direction. Axis-separable arccos computes per-feature angles and sums them — this gives the model per-feature sensitivity and allows feature weights to modulate the contribution of each feature independently. The axis-separable formulation showed consistent improvement over cosine similarity in ablation experiments.

### Why not learn the centroids?

Centroids are fixed at class means during training. Making centroids learnable adds K × 4F parameters (e.g. 4 × 44 = 176 for K=4, F=11) and introduces gradient interactions between centroid positions and branch weights. Empirically, centroid learning on small datasets showed the centroids drifting away from the class structure — the gradient couldn't maintain the relationship between centroid position and class membership. Post-training James-Stein shrinkage achieves the core goal (adjusting centroid positions for minority classes) without this instability.

### Why train/val split rather than full dataset for centroids?

The centroids used during training are computed from the training split only (after withholding the validation set). This creates a slight disadvantage: the model trains with N_train × 0.85 samples contributing to centroid estimation. The payoff is reliable early stopping — without a held-out validation set, early stopping fires essentially randomly, selecting checkpoints that happen to perform well on the training distribution rather than the test distribution.

### Training O(K) — Why this scales differently from XGBoost

XGBoost builds decision trees over all N samples at each iteration. AutoLambdaAVNN computes distances from mini-batches to K centroids. When N is large (200k+), XGBoost's O(N log N) per tree dominates while AutoLambdaAVNN's per-epoch cost stays fixed at O(N/batch × batch × K × F) = O(N × K × F). Since K << N, the model's effective computational scaling with N is much more favourable at large dataset sizes.

---

*Built iteratively over an extended research session. The AVNN family began as a static centroid classifier and evolved through learnable branch weights, four-branch geometry, FAISS acceleration, James-Stein shrinkage, diagonal Mahalanobis distance, ordinal EMD loss, and per-class temperature.*
