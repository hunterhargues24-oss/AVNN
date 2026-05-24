# AVNN: Step‑by‑Step Execution Order

This document explains the **exact order of operations** for the Angular Vector Nearest Neighbor (AVNN) classifier, focusing on the static version (`AVNNClassifier`) and the learnable version (`BranchAdaptiveAVNN`). All steps are deterministic and assume a standard sklearn‑style API.

---

## Part 1: Static AVNNClassifier (No Training)

The static model has **no training loop**. The `fit` method only computes centroids; `predict` performs all calculations per test point.

### Phase A: Fit (Store Training Information)

1. **Input validation**: Convert input arrays to NumPy float32 and ensure labels are 1D.
2. **Normalisation fit**:  
   - Compute per‑feature minima `mn` and ranges `rng = max(X) - min(X)` over the training set.  
   - Add a small epsilon to avoid division by zero.
3. **Normalise training data**:  
   `X_norm = (X - mn) / rng` → values in [0,1].  
   If `norm_range = '[-1,1]'`, apply `X_norm = -1.0 + 2.0 * X_norm` → values in [-1,1].
4. **Transform training data** (Euclidean branch):  
   `X_trans = tanh_ac(X_norm)` where:
   - For `[-1,1]` range: `tanh_ac(x) = tanh(arccos(x) * 0.8)`
   - For `[0,1]` range: `tanh_ac(x) = tanh(arccos(x) * 1.6)`
5. **Store training data** for KNN: `self.X_train_norm = X_norm`, `self.X_train_trans = X_trans`, `self.y_train = y`.
6. **Compute class centroids**:  
   For each class `c` in `np.unique(y)`:  
   - `self.centroids_norm_.append( X_norm[y == c].mean(axis=0) )`  
   - `self.centroids_trans_.append( X_trans[y == c].mean(axis=0) )`  
   Store as `(K, F)` arrays.

### Phase B: Predict (for a test point or batch)

#### Step 1 – Prepare test data
- Normalise test features using **training** `mn`, `rng` (same as step 3 above).  
- Apply the same `tanh_ac` transform (step 4).  
  Result: `X_norm_te`, `X_trans_te`.

#### Step 2 – AVM branch (centroid‑based)
- **Angular distance** to each centroid:
  ```
  theta_x = arccos(clip(X_norm_te, -1+eps, 1-eps))   # (N, F)
  theta_c = arccos(clip(centroids_norm, -1+eps, 1-eps)) # (K, F)
  ang_diff = abs(theta_x[:, None, :] - theta_c[None, :, :])   # (N, K, F)
  ang_dist = ang_diff.mean(axis=2)                            # (N, K)
  ```
- **Euclidean distance** in transformed space:
  ```
  euc_diff = X_trans_te[:, None, :] - centroids_trans[None, :, :]   # (N, K, F)
  euc_dist = sqrt( (euc_diff**2).sum(axis=2) + eps )               # (N, K)
  ```
- **Blended distance** (fixed α):
  ```
  d_cent = α * ang_dist + (1-α) * euc_dist
  ```
- **Inverse distance** and barycentric normalisation:
  ```
  raw_cent = 1.0 / (d_cent + eps)
  avm_proba = raw_cent / raw_cent.sum(axis=1, keepdims=True)
  ```

#### Step 3 – KNN branch (if λ < 1)
- **Full distance matrix** to training points:
  ```
  d_knn = α * ang_dist_to_train + (1-α) * euc_dist_to_train   # (N, N_train)
  ```
- For each test point `i`:
  - Find indices of `k` smallest distances (`np.argpartition`).
  - Weights `w = 1 / (d[i, idx] + eps)`, normalise.
  - For each neighbour `j` with weight `w_j`, add `w_j` to probability of class `y_train[j]`.
  - Result: `knn_proba` (N, K).

#### Step 4 – Blend AVM and KNN
```
final_proba = λ * avm_proba + (1-λ) * knn_proba
```

#### Step 5 – Prediction
```
predicted_class = classes_[argmax(final_proba, axis=1)]
```

---

## Part 2: Learnable BranchAdaptiveAVNN (PyTorch)

The learnable version introduces **training** for branch weights, feature weights, α, λ, τ. The KNN branch is **not used during training** (set λ = 1). All distances use the current learnable parameters.

### Phase A: Training (Fit)

#### Step 1 – Data split
- Stratified split into training and validation sets (early stopping based on validation macro F1).

#### Step 2 – Preprocessing (same as static)
- Compute min/max on training split, normalise, apply `tanh_ac`.

#### Step 3 – Compute fixed centroids (initial values)
- Class means in normalised and transformed spaces. These are **not** learnable – they serve as initial centroids for the AVM branch.

#### Step 4 – Build PyTorch model (`_Net`)
- **Learnable parameters**:
  - `raw_alpha` (sigmoid → α)
  - `raw_lambda` (sigmoid → λ) – kept but not used during training.
  - `log_tau` (exp → τ)
  - `raw_weights_ang` (softmax → per‑feature angular weights)
  - `raw_w_ang`, `raw_w_euc`, `raw_w_shape` (softmax → branch weights)
- **Non‑learnable buffers**:
  - Centroids (normalised and transformed)
  - Training data (normalised and transformed)
  - Training labels

#### Step 5 – Training loop
For each epoch:
  - Loop over batches (batch size 64).
  - For each batch:
    - Compute blended distance to **centroids** using current α, branch weights, feature weights.
    - Compute AVM probabilities with temperature τ.
    - Compute **class‑weighted NLL loss** with label smoothing.
    - Backpropagate, clip gradients, update parameters.
  - After epoch, evaluate **validation macro F1** (using pure AVM, λ=1).
  - Early stopping if no improvement for `patience` epochs.
  - Cosine annealing learning rate scheduler.

#### Step 6 – After training
- Store best parameters (branch weights, feature weights, α, λ, τ).
- Build FAISS index (if using fast inference) with **final branch weights**.

### Phase B: Inference (Predict)

After training, the model can be used for prediction with the same steps as the static version, but using the **learned branch weights**, **feature weights**, **α**, **λ**, **τ**. The KNN branch is now active (λ < 1). The FAISS index (if built) accelerates KNN.

---

## Summary Table of Operations

| Step | Static AVNN | Learnable BranchAdaptiveAVNN |
|------|-------------|------------------------------|
| Normalisation | Min‑max to [-1,1] | Same |
| Transform | `tanh_ac` (scale 0.8) | Same |
| Centroids | Means of training points | Same (fixed) |
| Branch weights | Fixed (e.g., 0.34/0.33/0.33) | Learned via softmax |
| Angular feature weights | Uniform (1.0 each) | Learned via softmax |
| α | Fixed (0.5) | Learned |
| λ | Fixed (0.7) | Learned |
| τ | None (temperature = 1) | Learned |
| KNN during training | Not applicable (no training) | Not used (λ=1) |
| KNN during inference | Exact (O(N²)) | FAISS‑accelerated |
| Training | None | Yes, with class weights, label smoothing, early stopping |

---

## Conclusion

The order of operations is **consistent across variants**: normalise, transform, compute distances (angular, Euclidean, optional shape), combine with branch weights, score via AVM and KNN, then blend. The static model uses fixed hyperparameters; the learnable model optimises them via pure AVM training. Both are deterministic and produce interpretable results. This modular design allows easy extension to new distance components or learning strategies.
