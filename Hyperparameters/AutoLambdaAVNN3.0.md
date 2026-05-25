# AutoLambdaAVNN — Hyperparameter Reference

Complete reference for all parameters with defaults, valid ranges, interactions, and tuning guidance. Parameters are grouped by the component they control.

---

## Quick-start configs

```python
# Balanced dataset (Iris, Wine cultivar, Breast Cancer)
balanced = dict(
    k=5, epochs=200, patience=50, val_fraction=0.15,
    label_smoothing=0.05, feat_weight_reg=0.05,
    centroid_reg=0.1, centroid_sep=0.05,
    mahalanobis=True, mahal_reg=0.01,
    ordinal=True, ordinal_weight=0.3, per_class_tau=True,
    lam_floor=0.5, auto_lambda=True,
    val_acc_weight=0.34, val_macro_weight=0.33, val_weighted_weight=0.33,
    use_ivf=False, random_state=42
)

# Moderate imbalance (2–10% minority — red wine, ArrivalType)
moderate_imbalance = dict(
    k=5, epochs=200, patience=60, val_fraction=0.15,
    label_smoothing=0.05, weight_cap=1.5, feat_weight_reg=0.01,
    centroid_reg=0.1, centroid_sep=0.05,
    mahalanobis=True, mahal_reg=0.05,
    ordinal=True, ordinal_weight=0.3, per_class_tau=True,
    lam_floor=0.5, auto_lambda=True,
    val_acc_weight=0.20, val_macro_weight=0.50, val_weighted_weight=0.30,
    use_ivf=False, random_state=42
)

# Severe imbalance (<2% minority)
severe_imbalance = dict(
    k=5, epochs=200, patience=60, val_fraction=0.15,
    label_smoothing=0.05, weight_cap=1.5, feat_weight_reg=0.01,
    centroid_reg=0.15, centroid_sep=0.05,
    mahalanobis=True, mahal_reg=0.10,
    ordinal=False, per_class_tau=False,
    lam_floor=0.5, auto_lambda=True,
    val_acc_weight=0.40, val_macro_weight=0.30, val_weighted_weight=0.30,
    use_ivf=False, random_state=42
)

# Large dataset (>50k samples — ArrivalType)
large_dataset = dict(
    k=10, epochs=20, patience=5, val_fraction=0.15,
    batch_size=256, label_smoothing=0.05, feat_weight_reg=0.01,
    centroid_reg=0.1, centroid_sep=0.05,
    mahalanobis=True, mahal_reg=0.01,
    ordinal=True, ordinal_weight=0.3, per_class_tau=True,
    lam_floor=0.5, auto_lambda=True,
    use_ivf=True, nlist=1000, nprobe=10,
    random_state=42, verbose=True
)
```

---

## Training

| Parameter | Default | Type | Range | Description |
|-----------|---------|------|-------|-------------|
| `k` | `5` | int | 1–50 | KNN neighbours used at inference. Higher k smooths the KNN branch but increases memory during FAISS search. Typical: 5 for small datasets, 10–20 for large. |
| `lr` | `1e-3` | float | 1e-4–1e-2 | Adam learning rate. Lower for noisy datasets, higher for clean well-separated ones. |
| `epochs` | `300` | int | 10–500 | Maximum training epochs. Early stopping usually fires well before this. Reduce to 20–50 for large datasets. |
| `batch_size` | `256` | int | 32–2048 | Mini-batch size. Larger batches give more stable gradients but reduce update frequency per epoch. 256 is optimal for most datasets. |
| `patience` | `60` | int | 10–100 | Early stopping patience in epochs after the composite val score stops improving. WARMUP of 20 epochs runs first — patience only counts after epoch 20. |
| `val_fraction` | `0.15` | float | 0.05–0.25 | Stratified validation holdout for early stopping and lambda search. Automatically enforced to minimum of `max(K*5, 5%)` to prevent degenerate splits on small imbalanced datasets. |
| `random_state` | `None` | int\|None | any | Seed for val split, DataLoader shuffle, and lambda search splits. Set for reproducibility. |
| `verbose` | `False` | bool | — | Print epoch-level val scores, branch weights, centroid drift, and lambda search results. |

---

## Regularisation

| Parameter | Default | Type | Range | Description |
|-----------|---------|------|-------|-------------|
| `weight_cap` | `1.5` | float | 1.0–5.0 | Maximum class weight in the NLL loss from sklearn's `balanced` strategy. Prevents extreme upweighting of tiny minority classes. At 1.0, all class weights are equal (no correction). |
| `weight_decay` | `1e-4` | float | 0–1e-2 | Adam L2 regularisation on all learnable parameters (branch weights, feature weights, tau, centroids). |
| `feat_weight_reg` | `0.05` | float | 0–0.5 | L2 penalty anchoring feature weights near uniform (1.0 each). Gradient at uniform = 0 so features only diverge when classification signal justifies it. Lower (0.01) for noisy labels where weak signals need to push through. Higher (0.1+) for small clean datasets. |
| `label_smoothing` | `0.05` | float | 0–0.2 | NLL label smoothing — replaces hard 0/1 targets with (ε/K, …, 1-ε+ε/K, …). Prevents overconfidence on training labels. |

---

## Learnable Centroids

Centroids are learnable `nn.Parameter` tensors initialised at class means. Backprop moves them simultaneously with branch and feature weights — the feature point and its class centroid are pulled toward each other every step.

| Parameter | Default | Type | Range | Description |
|-----------|---------|------|-------|-------------|
| `centroid_reg` | `0.1` | float | 0–1.0 | L2 anchor penalty — keeps centroids near initial class means. Prevents drift into meaningless positions. Higher = tighter anchor, less movement. Set to 0 to allow free centroid movement (risk of collapse). |
| `centroid_sep` | `0.05` | float | 0–0.5 | Separation penalty — maximises mean pairwise distance between centroids. Prevents all centroids collapsing toward the same point. Higher = centroids pushed further apart. |

**Interaction:** `centroid_reg` and `centroid_sep` oppose each other. The anchor pulls centroids toward their class means; separation pushes them apart. The NLL gradient pulls each centroid toward its class cluster. The three forces find an equilibrium at the best discriminative position.

**Reading centroid drift:** The summary shows how far each centroid moved from its initial class mean (in normalised feature space). Large drift on minority classes is expected and desirable — it indicates the centroid found a better discriminative position than the raw class mean. Near-zero drift on majority classes means the class mean was already well-positioned.

```
Centroid drift: class0=0.5599  class1=0.2757  class2=0.3711
# class0 (minority, 63 samples) moved most — class mean was unreliable
# class1 (majority, 1319 samples) moved least — class mean already stable
```

---

## Imbalance Handling

| Parameter | Default | Type | Range | Description |
|-----------|---------|------|-------|-------------|
| `mahalanobis` | `False` | bool | — | Enable diagonal Mahalanobis distance in AVM scoring. Each dimension scaled by per-class variance — tight clusters contribute more to distance, spread clusters contribute less. |
| `mahal_reg` | `1e-6` | float | 0–0.5 | Regularisation blending per-class variance toward global variance: `var_k = (1-reg)*var_k + reg*global_var`. Raise to 0.05–0.1 for small minority classes where variance estimate is noisy. |
| `ordinal` | `False` | bool | — | Enable Earth Mover's Distance (Wasserstein-1) ordinal loss. Only active when K > 2. Penalises mistakes proportional to ordinal distance — predicting class 0 when truth is class 2 costs more than predicting class 1. |
| `ordinal_weight` | `0.5` | float | 0–1.0 | Blend between NLL and EMD: `loss = (1-w)*NLL + w*EMD`. 0 = pure NLL, 1 = pure EMD. Recommended: 0.3 for mild ordinal structure, 0.5 for strong ordinal structure. |
| `per_class_tau` | `False` | bool | — | Each class centroid gets its own temperature parameter tau. Minority class centroids (noisy, few samples) learn higher tau (softer scoring). Majority class centroids learn lower tau (sharper scoring). Adds K parameters. |
| `shrinkage_strength` | `20.0` | float | 0–∞ | Legacy parameter — no longer has effect. Shrinkage was replaced by learnable centroids with `centroid_reg`. Kept for API compatibility. |

---

## Lambda (AVM/KNN Blend)

| Parameter | Default | Type | Range | Description |
|-----------|---------|------|-------|-------------|
| `lam_init` | `0.7` | float | 0–1.0 | Starting lambda. Used directly if `auto_lambda=False`. 1.0 = pure AVM (no FAISS needed). |
| `auto_lambda` | `True` | bool | — | Grid search for optimal AVM/KNN blend after training using the validation set and composite metric. |
| `lam_search_step` | `0.05` | float | 0.01–0.2 | Grid resolution for lambda search. 0.05 gives 11 candidates from `lam_floor` to 1.0. Finer step = more candidates = slightly better optimum but more compute. |
| `lam_floor` | `0.5` | float | 0–1.0 | Minimum lambda — AVM always has at least this weight in the blend. Prevents KNN from dominating and memorising training neighbours. Raise to 0.7+ on small datasets where KNN overfits severely. |
| `n_lam_splits` | `5` | int | 1–10 | Number of stratified splits for lambda search averaging. Kept for API compatibility — single-split search is currently used. |

**When to set `auto_lambda=False`:** Fixed lambda is appropriate when you want full reproducibility and have pre-tuned the value from a prior run. Also useful for ablation studies comparing pure AVM (lambda=1.0) vs blended performance.

---

## Validation Metric Weights

The composite validation metric `w_acc * acc + w_macro * F1_macro + w_weighted * F1_weighted` is used for both early stopping and lambda search. Weights should sum to 1.

| Parameter | Default | Balanced | Moderate imbalance | Severe imbalance |
|-----------|---------|----------|-------------------|-----------------|
| `val_acc_weight` | `0.33` | `0.34` | `0.20` | `0.40` |
| `val_macro_weight` | `0.34` | `0.33` | `0.50` | `0.30` |
| `val_weighted_weight` | `0.33` | `0.33` | `0.30` | `0.30` |

**Rationale:** On severely imbalanced data, macro F1 can be inflated by over-predicting the rare class. Weighting accuracy more heavily on severe imbalance prevents the model from selecting checkpoints that achieve high recall on the rare class by predicting it indiscriminately. On moderate imbalance, macro F1 deserves higher weight because minority class detection is the primary objective.

---

## FAISS Index

| Parameter | Default | Type | Range | Description |
|-----------|---------|------|-------|-------------|
| `use_ivf` | `True` | bool | — | Use IVF approximate index. Automatically falls back to Flat if construction fails (MemoryError) or dataset is too small (< 78 training samples). |
| `nlist` | `1000` | int | 1–10000 | IVF cluster count. Internally capped at `min(nlist, N//39, sqrt(N))` to prevent OOM. Rule of thumb: `sqrt(N)` is a safe upper bound. |
| `nprobe` | `10` | int | 1–nlist | Cells searched per query. Higher = more accurate, slower. Capped at nlist automatically. Typical: 10 for speed, 50–100 for accuracy-critical applications. |

**Memory guidance:**

| Training size | Recommended nlist | Peak FAISS memory |
|--------------|-------------------|------------------|
| < 10k | Flat (auto) | Minimal |
| 10k–50k | 100–200 | ~50MB |
| 50k–200k | 200–450 (sqrt cap) | ~200MB |
| > 200k | 400–1000 | 500MB+ |

---

## Inspectable Attributes (post-fit)

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `branch_weights_` | `(4,)` | Learned branch weights `[dir, mag, euc, shp]` summing to 1 |
| `feat_weights_` | `(F,)` | Learned per-feature weights (uniform = 1.0 each) |
| `tau_` | scalar or `(K,)` | Learned AVM temperature(s) — lower = sharper scoring |
| `lambda_` | scalar | Learned AVM/KNN blend (from auto search or `lam_init`) |
| `centroids_combined_` | `(K, 4F)` | Learned centroid positions in combined vector space |
| `centroid_drift_` | `(K,)` | Distance each centroid moved from initial class mean |
| `class_vars_` | `(K, 4F)` or None | Per-class per-dim variance for Mahalanobis (None if disabled) |
| `best_val_f1_` | scalar | Best composite val score achieved during training |
| `classes_` | `(K,)` | Class labels in order |

---

## Interactions and Gotchas

**`centroid_sep=0` with learnable centroids** — without separation pressure, centroids can collapse toward a common mean if the NLL gradient is weak (e.g. severe imbalance, noisy labels). Keep `centroid_sep > 0` unless you have a specific reason to remove it.

**`mahal_reg` and minority class size** — for a minority class with n_k < 50, the per-class variance estimate is unreliable. Set `mahal_reg` to 0.05–0.1 to borrow substantial global variance. For n_k > 200, `mahal_reg=0.01` is sufficient.

**`lam_floor=0.5` and small datasets** — on very small imbalanced datasets (< 100 minority samples), KNN memorises training neighbours and the lambda search rewards high KNN weight. `lam_floor=0.5` prevents lambda from collapsing to 0.3 or lower. Raise to 0.7 if you see lambda consistently finding 0.5 and performance is poor.

**`per_class_tau` adds K parameters** — on datasets with K=10+ classes and small minority counts, per-class tau can overfit. Consider `per_class_tau=False` for K > 5 unless minority classes are well-represented.

**`ordinal=True` requires ordered classes** — the EMD loss assumes class 0 < class 1 < class 2 in the integer encoding. Ensure your class encoding matches the natural ordering (e.g. low quality=0, medium=1, high=2). For non-ordinal multi-class problems, set `ordinal=False`.

**`val_fraction=0` behaviour** — triggers a minimum validation set of `max(K*5, 5% of data)` stratified by class. Do not set to 0 expecting to use all data for training — a validation set is always required for early stopping and lambda search.
