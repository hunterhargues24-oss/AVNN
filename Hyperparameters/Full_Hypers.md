# AVNN Hyperparameter Reference

---

## AVNNClassifier — Static (no training)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.5` | Angular vs Euclidean blend. `0` = pure Euclidean, `1` = pure angular. |
| `lam` | `0.7` | AVM centroid vs KNN blend. `1.0` = pure centroid, `0.0` = pure KNN. |
| `k` | `5` | Nearest neighbours for KNN branch. |
| `transform` | `'tanh_ac'` | Euclidean branch transform. Options: `tanh_ac`, `identity`, `tan`, `neglog`. |
| `norm_range` | `'[-1,1]'` | Normalisation range. `[-1,1]` = full semicircle, `[0,1]` = quarter-circle. |
| `eps` | `1e-10` | Division-by-zero guard. |

Centroids are computed as class means. No training required.

---

## BranchAdaptiveAVNN — Learnable (PyTorch)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | `5` | Nearest neighbours (inference only). |
| `lr` | `1e-3` | Adam learning rate. |
| `epochs` | `300` | Maximum training epochs. |
| `batch_size` | `64` | Mini-batch size. |
| `patience` | `60` | Early stopping patience on validation macro F1. |
| `val_fraction` | `0.15` | Validation holdout fraction (stratified). |
| `label_smoothing` | `0.05` | NLL loss label smoothing. |
| `weight_cap` | `1.5` | Class weight ceiling (balanced weights capped then renormalised). |
| `weight_decay` | `1e-4` | Adam L2 regularisation. |
| `norm_range` | `'[-1,1]'` | See static model. |
| `random_state` | `None` | Reproducibility seed. |
| `verbose` | `False` | Print epoch-level training progress. |

**Learned parameter initialisations:**

| Parameter | Init | Converged value |
|-----------|------|-----------------|
| `alpha` | `sigmoid(0.0)` → 0.500 | ~0.50 |
| `lambda` | `sigmoid(0.85)` → 0.700 | ~0.70 |
| `tau` | `exp(-0.5)` → 0.607 | ~0.61 |
| Branch weights | `softmax([0,0,0])` → 0.333 each | ~equal on balanced data |
| Feature weights | uniform → 1.0 each | sparse on imbalanced data |

---

## FastTriBranchAVNN — Static FAISS (no training)

Deploy this after extracting weights from a trained `BranchAdaptiveAVNN`,
or use the defaults for a no-training fast baseline.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lam` | `0.7` | AVM centroid vs KNN blend. `1.0` skips FAISS entirely. |
| `k` | `10` | Nearest neighbours (larger than static; FAISS makes this cheap). |
| `w_ang` | `0.34` | Angular branch weight. |
| `w_euc` | `0.33` | Euclidean branch weight. |
| `w_shape` | `0.33` | Shape branch weight. |
| `feat_weights` | `None` | Per-feature angular weights array. `None` = uniform. |
| `transform` | `'tanh_ac'` | See static model. |
| `norm_range` | `'[-1,1]'` | See static model. |
| `eps` | `1e-10` | Division-by-zero guard. |
| `use_ivf` | `True` | IVF approximate index (fast) vs flat exact index. |
| `nlist` | `1000` | IVF Voronoi cell count (capped at `n_train // 39`). |
| `nprobe` | `10` | Cells searched per query. Higher = more accurate, slower. |

---

## Recommended Defaults

Uniform branch weights worked well across all tested datasets.
For imbalanced datasets, the shape branch weight tends to drift higher (~0.42)
while angular dominates on structured categorical-like features.

| Dataset | `k` | `lam` | `w_ang` | `w_euc` | `w_shape` | Notes |
|---------|-----|-------|---------|---------|-----------|-------|
| Iris | 5 | 0.7 | 0.34 | 0.33 | 0.33 | Equal weights sufficient |
| Wine (cultivar) | 5 | 0.7 | 0.34 | 0.33 | 0.33 | Equal weights sufficient |
| Breast Cancer | 5 | 0.7 | 0.34 | 0.33 | 0.33 | Equal weights sufficient |
| Red Wine Quality | 5 | 0.7 | 0.38 | 0.28 | 0.34 | Angular slightly favoured |
| White Wine Quality | 5 | 0.7 | 0.24 | 0.34 | 0.42 | Shape favoured |
| ArrivalType (large) | 10 | 0.7 | 0.94 | 0.03 | 0.03 | Angular dominant; use FAISS IVF |

---

## Complexity Summary

| Model | Training | Inference | FAISS |
|-------|----------|-----------|-------|
| `AVNNClassifier` | None | O(n · F) | No |
| `BranchAdaptiveAVNN` | O(epochs · n · F) | O(n · F) | No |
| `FastTriBranchAVNN` | None | O(k · log n · F) | Yes |

`n` = training set size, `F` = number of features, `k` = neighbours.
