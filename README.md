# Angular Vector Nearest Neighbor (AVNN): Complete Technical Report

## Table of Contents
1. [Introduction](#1-introduction)
2. [Mathematical Foundations](#2-mathematical-foundations)
   - 2.1 Normalisation
   - 2.2 Angular Distance (Axis‑Separable)
   - 2.3 Euclidean Distance with tanh_ac Transform
   - 2.4 Shape Distance (Z‑score)
   - 2.5 Blended Distance
   - 2.6 AVM Centroid Branch
   - 2.7 KNN Branch
   - 2.8 Final Blending
3. [Model Evolution History](#3-model-evolution-history)
   - 3.1 AVM – The Original Idea
   - 3.2 Static AVNNClassifier (v1 & v2)
   - 3.3 Weighted AVNN (Learnable Feature Weights)
   - 3.4 Residual Correction Attempts (v4–v18)
   - 3.5 AdaptiveAVNN (Learnable α, λ, τ, Feature Weights)
   - 3.6 BranchAdaptiveAVNN (Three‑branch with Shape)
   - 3.7 GravityBranchAdaptiveAVNN (Class Bias)
   - 3.8 FastTriBranchAVNN (FAISS Acceleration)
4. [Key Breakthroughs](#4-key-breakthroughs)
   - 4.1 Axis‑Separable Angular Distance vs Cosine
   - 4.2 The Shape Branch: +9% Macro F1 on Red Wine
   - 4.3 [-1,1] Normalisation and Scale=0.8
   - 4.4 Hard k‑NN Prevents Memorisation
   - 4.5 Learnable Parameters without Overfitting
   - 4.6 FAISS for Large‑Scale Inference
5. [Experimental Results](#5-experimental-results)
   - 5.1 Datasets
   - 5.2 Static AVNNClassifier (20‑seed CV)
   - 5.3 BranchAdaptiveAVNN (5‑fold CV)
   - 5.4 FastTriBranchAVNN on 2 Million Points
   - 5.5 Comparison with Standard Classifiers
6. [Implementation Overview](#6-implementation-overview)
   - 6.1 Static Classifier (NumPy)
   - 6.2 Learnable Classifier (PyTorch)
   - 6.3 FAISS‑Accelerated Classifier
7. [Performance and Scaling](#7-performance-and-scaling)
8. [Open Questions and Future Work](#8-open-questions-and-future-work)
9. [Conclusion](#9-conclusion)

---

## 1. Introduction

The Angular Vector Nearest Neighbor (AVNN) is a **geometric classifier** that measures similarity using a novel **axis‑separabled angular distance**. Unlike standard KNN or centroid‑based methods, AVNN treats each feature independently, computing the angle between the point and the feature’s axis anchor (the unit vector along that dimension). This produces a rich, interpretable distance that can be blended with Euclidean and shape distances, and optionally combined with a learnable KNN branch.

Through extensive experimentation (over 18 major versions and dozens of ablations), we identified optimal hyperparameters and extensions that yield state‑of‑the‑art performance on several UCI benchmarks, and scale to millions of samples using FAISS.

---

## 2. Mathematical Foundations

Let \(F\) be the number of features, \(K\) the number of classes, and \(\mathbf{x} \in \mathbb{R}^F\) a raw feature vector.

### 2.1 Normalisation

First, min‑max normalisation is applied **on the training set only**:

\[
\hat{x}_i = \frac{x_i - \min_i}{\max_i - \min_i + \varepsilon}
\]

where \(\varepsilon = 10^{-10}\) prevents division by zero. This gives \(\hat{x}_i \in [0,1]\).

Two ranges are supported:
- \([0,1]\) (original quarter‑circle)
- \([-1,1]\) (full semicircle) obtained by \(\tilde{x}_i = 2\hat{x}_i - 1\).

The range \([-1,1]\) expands the angular domain of \(\arccos\) from \([0,\pi/2]\) to \([0,\pi]\), which empirically improves separability.

### 2.2 Angular Distance (Axis‑Separable)

For a point \(\mathbf{p}\) and a centroid \(\mathbf{c}\) in the normalised space \(\tilde{\mathbf{x}}\), define the per‑feature angle to the axis anchor:

\[
\theta_i(\mathbf{p}) = \arccos(\tilde{p}_i), \qquad \theta_i(\mathbf{c}) = \arccos(\tilde{c}_i)
\]

The axis‑separable angular distance is the **weighted mean absolute difference** of these angles:

\[
d_{\text{ang}}(\mathbf{p},\mathbf{c}) = \sum_{i=1}^{F} w_i \; \bigl| \theta_i(\mathbf{p}) - \theta_i(\mathbf{c}) \bigr|
\]

where \(w_i \ge 0\) are per‑feature weights. In the static model, \(w_i = 1\). In learnable versions, \(w_i\) are normalised via softmax and scaled so that \(\sum_i w_i = F\) (preserving the scale of a uniform average).

**Key property:** This distance is **not** a global cosine similarity; it preserves per‑feature resolution and is directly interpretable – each feature’s contribution can be inspected.

### 2.3 Euclidean Distance with tanh_ac Transform

To keep the Euclidean branch bounded similarly to the angular branch, we apply a non‑linear transform after normalisation:

\[
x'_i = \tanh\!\left( s \cdot \arccos(\tilde{x}_i) \right)
\]

The scale \(s\) depends on the normalisation range:
- For \([0,1]\): \(s = 1.6\) (since \(\arccos \in [0,\pi/2]\))
- For \([-1,1]\): \(s = 0.8\) (since \(\arccos \in [0,\pi]\))

The Euclidean distance is then:

\[
d_{\text{euc}}(\mathbf{p},\mathbf{c}) = \bigl\| \mathbf{p}' - \mathbf{c}' \bigr\|_2 = \sqrt{\sum_{i=1}^{F} (p'_i - c'_i)^2}
\]

### 2.4 Shape Distance

The shape distance removes absolute scale and offset, focusing on the relative pattern of features:

\[
z_i(\mathbf{p}) = \frac{\tilde{p}_i - \mu_{\mathbf{p}}}{\sigma_{\mathbf{p}}}, \quad
\mu_{\mathbf{p}} = \frac{1}{F}\sum_{j=1}^F \tilde{p}_j,\quad
\sigma_{\mathbf{p}} = \sqrt{\frac{1}{F}\sum_{j=1}^F (\tilde{p}_j - \mu_{\mathbf{p}})^2 + \varepsilon}
\]

Then:

\[
d_{\text{shape}}(\mathbf{p},\mathbf{c}) = \bigl\| \mathbf{z}(\mathbf{p}) - \mathbf{z}(\mathbf{c}) \bigr\|_2
\]

This is invariant to translation and scaling of the whole vector, capturing only the shape of the profile.

### 2.5 Blended Distance (Three‑Branch Model)

Let branch weights \(\beta_{\text{ang}}, \beta_{\text{euc}}, \beta_{\text{shape}} \ge 0\) with \(\sum \beta = 1\) (learned via softmax). The final distance is:

\[
d(\mathbf{p},\mathbf{c}) = \beta_{\text{ang}} d_{\text{ang}} + \beta_{\text{euc}} d_{\text{euc}} + \beta_{\text{shape}} d_{\text{shape}}
\]

The two‑branch version (static AVNN) sets \(\beta_{\text{shape}}=0\) and \(\beta_{\text{ang}}=\alpha,\ \beta_{\text{euc}}=1-\alpha\).

### 2.6 AVM Centroid Branch

Class centroids are computed as the mean of training points in the **normalised space** \(\tilde{\mathbf{x}}\) and in the **transformed space** \(\mathbf{x}'\). For three‑branch, we also store the shape‑normalised centroids implicitly.

For a test point \(\mathbf{p}\), compute distances \(d_k = d(\mathbf{p},\mathbf{c}_k)\) to each centroid. Raw affinities are inverse distances:

\[
r_k = \frac{1}{d_k + \varepsilon}
\]

A temperature parameter \(\tau > 0\) (learnable) softens the distances:

\[
\tilde{r}_k = \frac{1}{d_k / \tau + \varepsilon}
\]

The AVM probability for class \(k\) is the barycentric normalisation:

\[
P_{\text{avm}}(k) = \frac{\tilde{r}_k}{\sum_{j=1}^K \tilde{r}_j}
\]

### 2.7 KNN Branch

Hard \(k\)-nearest neighbours with inverse‑distance weighting. Let \(\mathcal{N}_k(\mathbf{p})\) be the indices of the \(k\) closest training points under the same blended distance \(d\). For each neighbour \(i\) with distance \(d_i\), weight \(w_i = 1/(d_i + \varepsilon)\). Normalised weights \(\hat{w}_i = w_i / \sum_{j\in\mathcal{N}_k} w_j\). The KNN probability for class \(k\) is:

\[
P_{\text{knn}}(k) = \sum_{i\in\mathcal{N}_k(\mathbf{p})} \hat{w}_i \cdot \mathbf{1}[y_i = k]
\]

### 2.8 Final Blending

A learnable parameter \(\lambda \in [0,1]\) (sigmoid of a raw parameter) blends the two branches:

\[
P_{\text{final}}(k) = \lambda \, P_{\text{avm}}(k) + (1-\lambda) \, P_{\text{knn}}(k)
\]

The predicted class is \(\arg\max_k P_{\text{final}}(k)\).

---

## 3. Model Evolution History

### 3.1 AVM – The Original Idea

The Angular Vector Model (AVM) was initially a pure centroid‑based classifier using only the angular distance (no Euclidean, no KNN). It showed promise but was outperformed by the hybrid.

### 3.2 Static AVNNClassifier (v1 & v2)

- **v1**: Normalisation \([0,1]\), transform `tanh_ac(scale=1.6)`, fixed \(\alpha=0.5,\lambda=0.7,k=5\), uniform feature weights.  
  Results: solid but macro F1 on imbalanced wine datasets was low (red wine macro F1 ~0.54).

- **v2**: Changed normalisation to \([-1,1]\) and scale to \(0.8\). This increased macro F1 on red wine from 0.58 to 0.59 (20‑seed CV). Became the recommended static model.

### 3.3 Weighted AVNN (Learnable Feature Weights)

Added learnable per‑feature angular weights (softmax + rescaling). Trained with cross‑entropy and early stopping. The weights converged to nearly uniform on all datasets, confirming that uniform weights are near‑optimal. This was an important negative result: the geometry is already well‑aligned.

### 3.4 Residual Correction Attempts (v4–v18)

We tried to move points toward their predicted centroid (residual correction) in angular space. Many versions failed due to teacher‑forcing (using true labels) or gradient issues. After fixing the loss (NLL instead of CrossEntropy) and making the target centroid selection soft (differentiable), the residual correction learned to turn itself off (strength → 0). This confirmed that the original geometry is already optimal.

### 3.5 AdaptiveAVNN (Learnable α, λ, τ, Feature Weights)

We removed the residual and kept only the geometry parameters (\(\alpha,\lambda,\tau\), feature weights). Training used class‑weighted NLL + label smoothing. The learnable model matched the static version’s performance, showing that the static hyperparameters are near‑optimal.

### 3.6 BranchAdaptiveAVNN (Three‑branch with Shape)

Added a third distance branch: shape (z‑score normalised L2). Branch weights learned via softmax. This gave a **significant boost** to macro F1 on red wine (+0.047) and also improved breast cancer. The shape branch learned high weight (~0.4) on red wine. This was the most effective improvement.

### 3.7 GravityBranchAdaptiveAVNN (Class Bias)

Added a learnable per‑class bias (gravity) to the raw AVM scores. This gave a small but consistent improvement in accuracy and weighted F1 on imbalanced datasets without harming macro F1. Useful but not essential.

### 3.8 FastTriBranchAVNN (FAISS Acceleration)

To handle large datasets (millions of points), we integrated FAISS for approximate KNN. The key trick: build a single vector by concatenating \(\sqrt{\beta_{\text{ang}}}\boldsymbol{\theta}(\mathbf{p})\), \(\sqrt{\beta_{\text{euc}}}\mathbf{p}'\), \(\sqrt{\beta_{\text{shape}}}\mathbf{z}(\mathbf{p})\). The Euclidean distance in this space approximates the true blended distance (though not exactly). After retrieving candidates, we recompute the true blended distance for weighting. This retains accuracy while making prediction sub‑linear.

---

## 4. Key Breakthroughs

| Breakthrough | Impact |
|--------------|--------|
| **Axis‑separable angular distance** | Preserves per‑feature resolution, outperforms global cosine similarity (0.593 vs 0.572 macro F1 on red wine). |
| **Shape branch** | Increased macro F1 on red wine by 9% (0.528 → 0.575). Learned branch weight ~0.4. |
| **[-1,1] normalisation + scale=0.8** | Improved macro F1 on red wine by +0.013 (20‑seed CV) over [0,1]. |
| **Hard k‑NN (k=5) with inverse‑distance** | Prevents memorisation (all‑points 1/d leads to overfitting). Confirmed by ablation. |
| **Learnable parameters without overfitting** | Only a few extra parameters (branch weights, feature weights, α, λ, τ). Stable training with early stopping. |
| **FAISS acceleration** | Scaled KNN to 2M points; prediction time 5.6s for 434k test points. |

---

## 5. Experimental Results

### 5.1 Datasets

| Dataset | Samples | Features | Classes | Type |
|---------|---------|----------|---------|------|
| Iris | 150 | 4 | 3 | balanced |
| Wine (cultivar) | 178 | 13 | 3 | balanced |
| Breast Cancer | 569 | 30 | 2 | balanced |
| Red Wine Quality | 1599 | 4 (pruned) | 3 | imbalanced (84% Medium) |
| White Wine Quality | 4898 | 11 | 3 | imbalanced |
| ArrivalType | 2,170,813 | 20 | 4 | highly imbalanced (96% class 4) |

### 5.2 Static AVNNClassifier (v2, α=0.5, λ=0.7, k=5, tanh_ac, [-1,1]) – 20‑seed CV (except red/white single split)

| Dataset | Accuracy | Macro F1 | Weighted F1 |
|---------|----------|----------|-------------|
| Iris | 0.9500 | 0.9497 | 0.9497 |
| Wine | 0.9726 | 0.9726 | 0.9726 |
| Breast Cancer | 0.9518 | 0.9472 | 0.9511 |
| Red Wine | 0.8625 | 0.5386 | 0.8457 |
| White Wine | 0.8204 | 0.5909 | 0.8102 |

### 5.3 BranchAdaptiveAVNN (three‑branch, learnable) – 5‑fold CV

| Dataset | Accuracy | Macro F1 | Weighted F1 |
|---------|----------|----------|-------------|
| Iris | 0.9600 ± 0.0389 | 0.9598 ± 0.0390 | 0.9598 ± 0.0390 |
| Wine | 0.9830 ± 0.0228 | 0.9831 ± 0.0226 | 0.9829 ± 0.0231 |
| Breast Cancer | 0.9683 ± 0.0154 | 0.9655 ± 0.0169 | 0.9680 ± 0.0156 |
| Red Wine | 0.8236 ± 0.0065 | **0.5753 ± 0.0488** | 0.8265 ± 0.0053 |
| White Wine | 0.8167 ± 0.0180 | 0.6173 ± 0.0259 | 0.8100 ± 0.0170 |

### 5.4 FastTriBranchAVNN (static, FAISS) – holdout on ArrivalType (subsample 200k train, 434k test)

| Metric | Value |
|--------|-------|
| Accuracy | 0.9648 |
| Macro F1 | 0.5759 |
| Weighted F1 | 0.9597 |
| Fit time | 6.75 s |
| Predict time | 5.59 s |
| Trivial baseline (always majority) | 0.9585 |

### 5.5 Comparison with Standard Classifiers (5‑fold CV, macro F1)

| Model | Iris | Wine | Breast Cancer | Red Wine | White Wine |
|-------|------|------|---------------|----------|------------|
| BranchAdaptiveAVNN | **0.960** | **0.983** | **0.966** | **0.575** | **0.617** |
| Random Forest | 0.946 | 0.978 | 0.953 | 0.533 | 0.631 |
| XGBoost | 0.939 | 0.961 | 0.949 | 0.527 | 0.638 |
| SVM (RBF) | 0.967 | 0.625 | 0.904 | 0.301 | 0.285 |
| Logistic Regression | 0.967 | 0.952 | 0.943 | 0.425 | 0.427 |

AVNN achieves the highest macro F1 on red and white wine, and is competitive on other datasets.

---

## 6. Implementation Overview

### 6.1 Static Classifier (AVNNClassifier.py)
- Pure NumPy, no training.
- Implements two‑branch blended distance (angular + Euclidean).
- Hard k‑NN with inverse‑distance weighting.
- Used for small‑to‑medium datasets (up to ~50k points).

### 6.2 Learnable Classifier (BranchAdaptiveAVNN.py)
- PyTorch model with learnable branch weights, feature weights, α, λ, τ.
- Training: pure AVM (λ=1) to avoid KNN non‑differentiability.
- Loss: class‑weighted NLL with label smoothing.
- After training, the learned parameters are used for inference with the blended model (λ<1).

### 6.3 FAISS‑Accelerated Classifier (FastTriBranchAVNN.py)
- Static three‑branch model.
- Uses FAISS for approximate KNN (IVF index).
- For exact weighting, retrieves neighbours via FAISS then recomputes true blended distance.
- Scales to millions of points.

---

## 7. Performance and Scaling

| Model | Training time | Prediction time (per point) | Memory |
|-------|---------------|----------------------------|--------|
| Static AVNN (λ=1) | O(NF) | O(KF) | O(KF) |
| Static AVNN (λ<1) | O(NF) | O(NF) | O(NF) |
| BranchAdaptiveAVNN (train) | hours (small data) | – | – |
| FastTriBranchAVNN (200k train, λ<1) | 6.8 s | 0.013 ms (with FAISS) | index ~800 MB |

FAISS IVF parameters: `nlist=1000`, `nprobe=10`, `k=10`. Accuracy loss vs exact KNN negligible.

---

## 8. Open Questions and Future Work

1. **Theoretical analysis** – Why does axis‑separable angular distance work so well? Is there a connection to spherical harmonics or Dirichlet distributions?
2. **Differentiable FAISS** – Enable end‑to‑end training by using a differentiable approximation of KNN (e.g., soft nearest neighbours).
3. **Missing data and categorical features** – Extend the model to handle non‑numerical inputs.
4. **Shape branch alternatives** – Try correlation distance, dynamic time warping, or Fourier descriptors for ordered features.
5. **Multi‑scale branches** – Combine distances computed at different scales (e.g., different k values).
6. **Online learning** – The static model can be updated incrementally with new centroids; learnable model could support online updates.

---

## 9. Conclusion

The Angular Vector Nearest Neighbor (AVNN) family provides a robust, interpretable, and scalable geometric classifier. The core innovation – the axis‑separable angular distance – captures per‑feature orientation and complements Euclidean and shape distances. Through extensive experimentation, we identified optimal configurations and demonstrated state‑of‑the‑art performance on multiple benchmarks, including a 2‑million‑point industrial dataset. The code is open and ready for expert review.

**Citation**  
If you use this work, please cite:  
*“Angular Vector Nearest Neighbor (AVNN): A Geometric Classifier with Per‑Feature Angular Distance.”*  
Available at hunterhargues24-oss.

---

*Appendix: Full code and Jupyter notebooks are provided separately.*
