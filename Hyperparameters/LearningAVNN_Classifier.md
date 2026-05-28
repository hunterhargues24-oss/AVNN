# LearningAVNN — Hyperparameter Reference

37 parameters total. Defaults are conservative and work for most balanced datasets.
Imbalanced datasets require targeted adjustments — see the **Imbalance** section.

---

## Training

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `k` | 5 | 3–20 | KNN neighbours. Increase for large datasets or severe imbalance. |
| `lr` | 1e-3 | 1e-4 – 5e-3 | Adam learning rate. Lower for noisy/small datasets. |
| `epochs` | 300 | 100–500 | Max training epochs. Early stopping usually fires well before this. |
| `batch_size` | 256 | 64–1024 | Mini-batch size. Larger batches give smoother gradients; smaller expose minority samples more frequently. |
| `patience` | 60 | 20–100 | Early stopping patience in epochs. Lower for fast experiments, higher for slow-converging imbalanced datasets. |
| `val_fraction` | 0.15 | 0.05–0.25 | Fraction held out for early stopping and lambda search. Lower for very small datasets. |
| `random_state` | None | int | Seed for reproducibility. |
| `verbose` | False | bool | Prints epoch-level branch weights and val F1. |

---

## Regularisation

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `label_smoothing` | 0.05 | 0.0–0.15 | NLL label smoothing. Prevents overconfident predictions. Reduce to 0.0 for clean labels. |
| `weight_cap` | 1.5 | 1.0–5.0 | Maximum class weight for imbalance correction. Higher values emphasise minority classes more aggressively. |
| `weight_decay` | 1e-4 | 0–1e-3 | Adam L2 regularisation on all parameters. |
| `feat_weight_reg` | 0.05 | 0.0–0.2 | Pulls per-feature weights toward 1.0. Use 0.01 for imbalanced datasets where feature discrimination matters more. |
| `ortho_reg` | 0.01 | 0.0–0.1 | Gram matrix penalty that decorrelates the six branches. Encourages each branch to capture independent signal. |

---

## Prototypes

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `centroid_reg` | 0.1 | 0.0–0.5 | Anchors prototypes near their initial class means. Higher values keep prototypes stable; lower allows more drift. |
| `centroid_sep` | 0.05 | 0.0–0.2 | Pushes class prototypes apart. Helps when classes are close together in feature space. |
| `n_prototypes` | 1 | 1–5 | Prototypes per class. Use >1 for multimodal class distributions (e.g. a quality class that contains two distinct sub-populations). |
| `intra_sep` | 0.05 | 0.0–0.2 | Pushes within-class prototypes apart. Only active when `n_prototypes > 1`. |

---

## Triangle Scoring

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `use_triangle` | False | bool | Enables pairwise deviation cross-product scoring. Proven +0.058 macro F1 on moderate imbalance (4/82/14%). Neutral on well-separated balanced data. Avoid for F > 30 (too many pairs). |
| `tri_weight` | 0.5 | 0.1–0.7 | Blend weight: `(1 - w)·AVM + w·triangle`. Use 0.2–0.3 for imbalanced real data. 0.5 for synthetic clean data. |
| `use_dual_boundary` | False | bool | Extends boundary branch from F to 2F by adding distances to negative face-centres. Adds cross-feature coupling signal for features where low values are discriminative. |

---

## Distance Metric

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `mahalanobis` | True | bool | Full RDA Mahalanobis at inference. Accounts for cluster shape and orientation. Almost always helps — only disable for very high-dimensional data (F > 100). |
| `mahal_reg` | 1e-6 | 1e-8–1e-4 | Ridge regularisation on covariance matrix. Increase if Cholesky decomposition fails (rare). |
| `mahal_alpha` | 0.5 | 0.0–1.0 | Blend between per-class QDA (0.0) and pooled LDA (1.0). Lower for datasets where classes have very different shapes. Use 0.2–0.3 for severe imbalance where minority classes have few samples. |

---

## Lambda Gate

Controls how AVM and KNN predictions are blended at inference.

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `lam_floor` | 0.5 | 0.3–0.9 | Minimum AVM weight in the scalar lambda grid search. Lambda is never searched below this value. |
| `entropy_lambda` | False | bool | Legacy entropy gate — uses AVM entropy only to compute per-sample lambda. Superseded by the confidence gate (set `lambda_=None` automatically when confidence gate wins). |
| `lam_confident` | 0.90 | 0.7–1.0 | Maximum lambda when AVM is confident (low entropy). |
| `lam_uncertain` | 0.40 | 0.2–0.6 | Minimum lambda when AVM is uncertain (high entropy). |

**How lambda is chosen:** After training, both a scalar grid search (from `lam_floor` to 1.0 in 0.05 steps) and the confidence gate are evaluated on the validation set. Whichever scores higher wins. `lambda_=None` in the fitted model means the confidence gate won.

---

## Imbalance Handling

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `ordinal` | False | bool | Ordinal EMD loss — penalises misclassifications proportionally to class distance. Use when classes have a natural ordering (e.g. quality scores 1–5). |
| `ordinal_weight` | 0.5 | 0.1–0.7 | Blend weight for ordinal loss alongside NLL. |
| `supcon` | False | bool | Supervised Contrastive loss — pulls same-class embeddings together in AVM space. Helps minority classes cluster more tightly. Requires at least 2 minority samples per batch to be effective. |
| `supcon_weight` | 0.3 | 0.1–0.5 | Blend weight for SupCon loss. |
| `supcon_temp` | 0.1 | 0.05–0.5 | SupCon temperature. Lower = sharper contrastive separation. |
| `ema_centroids` | False | bool | EMA centroid stabilisation — smooths prototype updates using exponential moving average. Critical for minority classes with sparse gradient signal. |
| `ema_beta` | 0.9 | 0.8–0.99 | EMA decay rate. Higher = slower centroid movement, more stable. |
| `per_class_tau` | True | bool | Per-class temperature scaling. Allows each class to have different scoring sharpness. Helps minority classes avoid being dominated by majority class confidence. |
| `val_macro_bias` | 0.5 | 0.0–1.0 | Controls early stopping criterion. 0.0 = pure accuracy. 1.0 = pure macro F1. Use 0.75–0.9 for imbalanced datasets to ensure early stopping optimises for minority class recovery. |

---

## FAISS Index

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `use_ivf` | True | bool | IVF approximate index for fast KNN on large datasets. Automatically falls back to FlatL2 (exact) for small datasets. Disable for datasets under ~1000 training samples. |

---

## Recommended Configs by Use Case

### Balanced dataset, clean labels
```python
LearningAVNN(
    k=5, epochs=300,
    use_triangle=False,
    mahalanobis=True, mahal_alpha=0.5,
    val_macro_bias=0.5,
)
```

### Moderate imbalance (2–15% minority)
```python
LearningAVNN(
    k=5, epochs=300,
    ordinal=True, ordinal_weight=0.3,
    supcon=True, supcon_weight=0.3,
    ema_centroids=True,
    use_triangle=True, tri_weight=0.3,
    feat_weight_reg=0.01,
    val_macro_bias=0.8,
    mahal_alpha=0.5,
)
```

### Severe imbalance (<2% minority)
```python
LearningAVNN(
    k=10, epochs=300, patience=80,
    ema_centroids=True, ema_beta=0.95,
    use_triangle=True, tri_weight=0.3,
    mahal_alpha=0.2,
    val_macro_bias=0.9,
    feat_weight_reg=0.01,
    use_ivf=True,
)
```

### Large dataset (>50k samples)
```python
LearningAVNN(
    k=10, batch_size=512,
    epochs=200, patience=40, val_fraction=0.05,
    use_ivf=True,
    mahal_alpha=0.3,
    val_macro_bias=0.9,
)
```
