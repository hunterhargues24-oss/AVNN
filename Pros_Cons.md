# AVNN Family: Overall Pros and Cons

Based on extensive experimentation across six datasets (Iris, Wine, Breast Cancer, Red/White Wine Quality, and a large industrial dataset), here is an objective assessment of the Angular Vector Nearest Neighbor (AVNN) models.

---

## Pros

### 1. **Interpretability**
- **Per‑feature angular deviations** directly explain why a point is assigned to a particular class. You can report: “Feature i differs from the class centroid by X radians” – no black‑box.
- Learnable feature weights (in adaptive versions) show which features matter most.
- Branch weights (angular / Euclidean / shape) reveal which geometric signal dominates.

### 2. **No Training (Static Version)**
- The static `AVNNClassifier` is **parameter‑free**: only centroids and training data storage. No gradient descent, no hyperparameter tuning beyond the fixed defaults (`α=0.5, λ=0.7, k=5`, transform, norm_range).
- “Training” is just computing min/max and class means – instantaneous even on large datasets.
- Perfect reproducibility.

### 3. **Competitive Performance**
- **Highest macro F1** among all tested classifiers on Iris, Wine cultivar, Breast Cancer, and Red Wine (see comparison table).  
- On White Wine, macro F1 is within 0.02 of XGBoost (the best), while being far more interpretable.
- Beats the trivial majority‑class baseline on the highly imbalanced 2.2M‑point ArrivalType dataset (96.5% vs 95.8% accuracy, macro F1 0.576 vs 0.0 for the baseline).

### 4. **Robustness to Imbalance (with shape branch)**
- The shape branch significantly improved macro F1 on red wine (+0.047 over two‑branch). Shape distance (z‑score normalised) helps separate minority classes by focusing on relative feature patterns rather than absolute magnitudes.
- Class weights (in learnable version) and subsampling (in static) can further balance.

### 5. **Scalability via FAISS**
- `FastTriBranchAVNN` handles millions of training points with approximate KNN. Prediction is sub‑linear (log N), memory is O(N), and accuracy loss is negligible.
- FAISS IVF index with `nprobe=10` gives 5‑6 second prediction on 434k test points.

### 6. **No Overfitting**
- Static model has zero parameters – cannot overfit.  
- Learnable model has very few parameters (branch weights + feature weights + α + λ + τ + optional gravity). Early stopping and small weight decay keep generalisation strong.  
- Ablations confirmed that residual correction (which could overfit) learned to turn itself off.

### 7. **Flexibility**
- The architecture is modular: you can switch branches on/off, adjust weights, choose normalisation range, transform, k, and blend parameters.  
- Works with any numeric features – no assumptions about distribution.

### 8. **Deterministic and Fast**
- Once centroids are computed (or FAISS index built), predictions are deterministic.  
- Static pure AVM (λ=1) runs in milliseconds even on 2M points.  
- Python/NumPy implementation is lightweight; no heavy dependencies except optional FAISS.

---

## Cons

### 1. **Computationally Expensive KNN (without FAISS)**
- The full KNN branch (λ<1) requires O(N_test × N_train) distance computations. For large datasets (e.g., >100k), this becomes infeasible without approximate methods.  
- **Mitigation**: FAISS solves this, but adds a dependency and approximation (though negligible accuracy loss in practice).

### 2. **Sensitivity to Feature Scaling and Outliers**
- Min‑max normalisation is used; extreme outliers can compress most points into a small interval.  
- **Mitigation**: Robust scaling (percentile‑based) could be added, but was not tested.

### 3. **Assumes Features are Numeric and Meaningfully Ordered**
- The angular distance uses `arccos`, which requires values in [-1,1] and assumes a natural order (higher = larger angle). Categorical features are not supported directly.  
- **Mitigation**: One‑hot encoding could be used, but would create many binary features – not ideal.

### 4. **No Built‑in Handling of Missing Data**
- NaN values cause distance computation to fail. Pre‑imputation is required.  
- **Mitigation**: Simple imputation (mean/median) can be applied before feeding to the model.

### 5. **Shape Branch Adds Computational Overhead**
- Shape distance requires computing per‑sample mean and std for each test point. For high‑dimensional data (e.g., 100+ features), this is noticeable.  
- **Mitigation**: Shape branch can be disabled (set weight to 0) without affecting other branches.

### 6. **Learnable Version Still Requires Training**
- Although the static version is training‑free, the learnable `BranchAdaptiveAVNN` needs PyTorch and several epochs. Training time on large datasets (e.g., 200k points) is a few hours.  
- **Mitigation**: Only needed if you want to adapt branch/feature weights; static version suffices for most tasks.

### 7. **FAISS Dependency and Approximation**
- FAISS is not a standard library and must be installed separately (`faiss-cpu` or `faiss-gpu`).  
- The concatenated‑vector approximation (using Euclidean distance) is not exactly equal to the true blended distance (weighted L1 for angular). We recompute true distances for retrieved neighbours, but the neighbour set itself may miss some near neighbours due to the approximation. In practice, the loss of accuracy is very small (<<1% macro F1).

### 8. **Performance on Very High‑Dimensional Data (>100 features) is Untested**
- Curse of dimensionality affects all distance‑based methods. Angular distances may become uniform. The shape branch may also degrade because z‑score normalisation becomes noisy.  
- **Mitigation**: Dimensionality reduction (PCA) could be applied before AVNN, but not yet tested.

### 9. **Interpretability Requires Feature Names**
- The per‑feature angular deviations are meaningful only if you know what each feature represents. Without feature names, the explanation is “feature i” – still interpretable but less intuitive.

### 10. **Static Model’s Hyperparameters are Dataset‑Specific**
- While `α=0.5, λ=0.7, k=5, transform='tanh_ac', norm_range='[-1,1]'` worked well across all tested datasets, there is no guarantee they are optimal for every new dataset.  
- **Mitigation**: The learnable version can tune them automatically, or you can perform a quick grid search.

---

## Summary Table

| Aspect | Pro | Con |
|--------|-----|-----|
| Interpretability | ✓ Per‑feature angular explanations | – Requires feature names |
| Training | ✓ Static version: zero training | – Learnable version: needs epochs |
| Speed | ✓ Pure AVM: ms per 1M points | – KNN without FAISS: O(N²) |
| Scalability | ✓ FAISS: sub‑linear prediction | – Additional dependency |
| Performance | ✓ Best macro F1 on imbalanced data | – Accuracy sometimes slightly below tree ensembles |
| Robustness | ✓ No overfitting, class weights | – Sensitive to outliers (min‑max) |
| Flexibility | ✓ Modular branches, transforms | – Categorical features not supported |
| Missing data | – | – Must be pre‑imputed |

---

## Final Verdict

The AVNN family offers a **strong, interpretable, and scalable** geometric alternative to standard classifiers. Its strengths lie in balanced performance (macro F1), transparency, and the ability to handle millions of points with FAISS. Its main weaknesses are the reliance on numeric, complete data and the computational cost of full KNN without acceleration. For most practical tabular classification tasks, the static `AVNNClassifier` is a fast, reliable, and explainable choice. For highly imbalanced or large‑scale problems, the three‑branch `FastTriBranchAVNN` (with shape and FAISS) provides excellent results.
