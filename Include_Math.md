# Deep Technical Breakdown of AVNN Mathematics

> Mathematical specification for the Angular Vector Nearest Neighbor (AVNN) family of models, including static, learnable, and FAISS-accelerated variants.

---

# 1. Overview

The Angular Vector Nearest Neighbor (AVNN) framework is a geometric classification architecture built around:

* Axis-separable angular geometry
* Euclidean geometry in transformed angular space
* Optional shape-invariant geometry
* Centroid-based probabilistic inference (AVM branch)
* Weighted k-nearest-neighbor voting (KNN branch)

Unlike cosine similarity, AVNN treats each feature as an independent angular axis and measures per-feature angular deviation directly.

The framework supports both:

* **Static parameter-free models**
* **Learnable differentiable models trained with backpropagation**

---

# 2. Feature Normalization

Let:

[
\mathbf{x} \in \mathbb{R}^{F}
]

be a feature vector with (F) features.

Normalization is computed using training-set statistics only.

---

## 2.1 Standard Min-Max Scaling to ([0,1])

[
\hat{x}_i =
\frac{x_i - \min_i}
{\max_i - \min_i + \varepsilon}
]

where:

* (\min_i) = minimum value of feature (i)
* (\max_i) = maximum value of feature (i)
* (\varepsilon = 10^{-10})

This produces:

[
\hat{x}_i \in [0,1]
]

---

## 2.2 Recommended Scaling to ([-1,1])

AVNN experimentally performs better using full semicircular angular space:

[
\tilde{x}_i = 2\hat{x}_i - 1
]

giving:

[
\tilde{x}_i \in [-1,1]
]

This expands the angular domain:

[
\arccos(\tilde{x}_i)
\in [0,\pi]
]

instead of:

[
[0,\pi/2]
]

which empirically improves separability.

---

# 3. Angular Feature Transform

The Euclidean branch operates on a bounded non-linear angular transform:

[
x_i' =
\tanh\left(
s \cdot \arccos(\tilde{x}_i)
\right)
]

where:

* (s) = scaling factor

Recommended values:

| Input Range | Angular Domain | Recommended (s) |
| ----------- | -------------- | --------------- |
| ([0,1])     | ([0,\pi/2])    | 1.6             |
| ([-1,1])    | ([0,\pi])      | 0.8             |

This scaling avoids saturation in the (\tanh) branch.

---

# 4. Axis-Separable Angular Geometry

AVNN does **not** use standard cosine similarity.

Instead, each feature independently contributes an angular displacement.

For feature (i):

[
\theta_i(\mathbf{p}) =
\arccos(\tilde{p}_i)
]

For centroid (\mathbf{c}):

[
\theta_i(\mathbf{c}) =
\arccos(\tilde{c}_i)
]

The angular distance is:

[
d_{\text{ang}}(\mathbf{p},\mathbf{c})
=====================================

\sum_{i=1}^{F}
w_i
\left|
\theta_i(\mathbf{p})
--------------------

\theta_i(\mathbf{c})
\right|
]

where:

[
w_i \ge 0
]

are per-feature weights.

---

## 4.1 Feature Weight Normalization

In learnable variants:

[
w_i =
F \cdot
\frac{
e^{a_i}
}{
\sum_j e^{a_j}
}
]

where:

* (a_i) are raw learnable logits
* weights sum to (F)

This preserves the scale of a uniform average.

---

# 5. Euclidean Geometry

Euclidean distance is computed in transformed angular space:

[
d_{\text{euc}}(\mathbf{p},\mathbf{c})
=====================================

\left|
\mathbf{p}'
-----------

\mathbf{c}'
\right|_2
]

Expanded:

[
d_{\text{euc}}
==============

\sqrt{
\sum_{i=1}^{F}
(p_i' - c_i')^2
}
]

---

# 6. Shape Geometry

The optional shape branch removes global scale and offset information.

Define:

[
\mu_{\mathbf{p}}
================

\frac{1}{F}
\sum_{i=1}^{F}
\tilde{p}_i
]

[
\sigma_{\mathbf{p}}
===================

\sqrt{
\frac{1}{F}
\sum_{i=1}^{F}
(\tilde{p}*i - \mu*{\mathbf{p}})^2
+
\varepsilon
}
]

Then:

[
z_i(\mathbf{p})
===============

\frac{
\tilde{p}*i - \mu*{\mathbf{p}}
}{
\sigma_{\mathbf{p}}
}
]

Shape distance:

[
d_{\text{shape}}(\mathbf{p},\mathbf{c})
=======================================

\left|
\mathbf{z}(\mathbf{p})
----------------------

\mathbf{z}(\mathbf{c})
\right|_2
]

This captures feature-pattern similarity independent of scale.

---

# 7. Blended Distance Function

AVNN combines multiple geometric branches.

For branch weights:

[
\beta_{\text{ang}},
\beta_{\text{euc}},
\beta_{\text{shape}}
\ge 0
]

with:

[
\beta_{\text{ang}}
+
\beta_{\text{euc}}
+
\beta_{\text{shape}}
= 1
]

the full blended distance becomes:

[
d(\mathbf{p},\mathbf{c})
========================

\beta_{\text{ang}}
d_{\text{ang}}
+
\beta_{\text{euc}}
d_{\text{euc}}
+
\beta_{\text{shape}}
d_{\text{shape}}
]

---

## 7.1 Two-Branch AVNN

The original AVNN formulation uses:

[
\beta_{\text{shape}} = 0
]

and:

[
\beta_{\text{ang}} = \alpha
]

[
\beta_{\text{euc}} = 1-\alpha
]

giving:

[
d =
\alpha d_{\text{ang}}
+
(1-\alpha)d_{\text{euc}}
]

---

# 8. AVM Centroid Branch

Each class centroid is computed from training data.

For class (k):

[
\mathbf{c}_k
============

\frac{1}{N_k}
\sum_{n:y_n=k}
\mathbf{x}_n
]

Distances to centroids:

[
d_k = d(\mathbf{p},\mathbf{c}_k)
]

Raw inverse-distance affinity:

[
r_k =
\frac{1}{d_k + \varepsilon}
]

Temperature-scaled affinity:

[
\tilde{r}_k =
\frac{1}{d_k/\tau + \varepsilon}
]

Final AVM probability:

[
P_{\text{avm}}(k)
=================

\frac{
\tilde{r}_k
}{
\sum_j \tilde{r}_j
}
]

---

# 9. KNN Branch

AVNN uses hard k-nearest neighbors with inverse-distance weighting.

Let:

[
\mathcal{N}_k(\mathbf{p})
]

be the set of (k) nearest neighbors.

Neighbor weight:

[
w_i =
\frac{1}{d_i + \varepsilon}
]

Normalized:

[
\hat{w}*i =
\frac{
w_i
}{
\sum*{j \in \mathcal{N}_k} w_j
}
]

Class probability:

[
P_{\text{knn}}(c)
=================

\sum_{i \in \mathcal{N}_k}
\hat{w}_i
\mathbf{1}[y_i=c]
]

---

# 10. Final Prediction

The AVM and KNN branches are blended:

[
P_{\text{final}}(k)
===================

\lambda P_{\text{avm}}(k)
+
(1-\lambda)P_{\text{knn}}(k)
]

where:

[
\lambda \in [0,1]
]

is learned via sigmoid:

[
\lambda = \sigma(a_\lambda)
]

Prediction:

[
\hat{y}
=======

\arg\max_k P_{\text{final}}(k)
]

---

# 11. Residual Correction Branch

Learnable AVNN variants introduce geometric residual correction.

The model predicts a soft centroid assignment:

[
p_k =
\text{softmax}
\left(
-\frac{d_k}{\tau}
\right)
]

Soft centroid target:

[
\mathbf{c}_{\text{soft}}
========================

\sum_k p_k \mathbf{c}_k
]

Residual correction:

[
\Delta
======

g \beta
\mathbf{s}
\odot
(\mathbf{c}_{\text{soft}} - \mathbf{x})
]

where:

* (g = \sigma(a_g)) = residual gate
* (\beta = e^{b}) = residual magnitude
* (\mathbf{s} = \tanh(\mathbf{u})) = feature scaling vector
* (\odot) = elementwise multiplication

Corrected feature vector:

[
\mathbf{x}_{\text{corr}}
========================

\mathbf{x}
+
\Delta
]

This replaces earlier teacher-forced centroid correction and removes train/inference mismatch.

---

# 12. Loss Function

The corrected learnable AVNN uses:

[
\text{NLLLoss}
]

on log-probabilities.

---

## 12.1 Negative Log Likelihood

[
\mathcal{L}_{\text{NLL}}
========================

-\log P(y)
]

---

## 12.2 Label Smoothing

With smoothing parameter (\epsilon):

[
\mathcal{L}_{\text{smooth}}
===========================

(1-\epsilon)
\mathcal{L}*{\text{NLL}}
+
\frac{\epsilon}{K}
\sum*{k=1}^{K}
-\log P(k)
]

Recommended:

[
\epsilon = 0.05
]

---

## 12.3 Class Weighting

For class (c):

[
v_c
===

\frac{N}{K \cdot n_c}
]

where:

* (N) = total samples
* (K) = number of classes
* (n_c) = samples in class (c)

Weights are capped and renormalized.

---

# 13. FAISS Approximation

To accelerate neighbor retrieval, AVNN embeds all branches into a single vector.

Define:

[
\mathbf{v}(\mathbf{p})
======================

\left[
\sqrt{\beta_{\text{ang}}},
\boldsymbol{\theta}(\mathbf{p}),
;
\sqrt{\beta_{\text{euc}}},
\mathbf{p}',
;
\sqrt{\beta_{\text{shape}}},
\mathbf{z}(\mathbf{p})
\right]
]

Then:

[
|
\mathbf{v}(\mathbf{p})
----------------------

\mathbf{v}(\mathbf{c})
|_2^2
]

approximates the blended AVNN distance.

This is approximate because the angular branch uses weighted L1 geometry.

After FAISS retrieval, true AVNN distance is recomputed.

---

# 14. Static AVNN Configuration

The best-performing static configuration experimentally observed:

| Parameter               | Value    |
| ----------------------- | -------- |
| Normalization           | ([-1,1]) |
| Transform scale         | 0.8      |
| Angular weight (\alpha) | 0.5      |
| AVM/KNN blend (\lambda) | 0.7      |
| Neighbors (k)           | 5        |

---

# 15. Empirical Findings

## 15.1 Key Architectural Discoveries

### Full Semicircle Improves Geometry

Using:

[
[-1,1]
]

instead of:

[
[0,1]
]

expands angular coverage from:

[
[0,\pi/2]
\rightarrow
[0,\pi]
]

and improves baseline separability.

---

### Saturation Correction Was Critical

Using:

[
\tanh(1.6 \cdot \arccos(x))
]

on:

[
[-1,1]
]

caused Euclidean branch saturation.

Reducing scale to:

[
0.8
]

restored useful Euclidean geometry.

---

### KNN Branch Was Essential

Ablation experiments showed:

[
\Delta \text{F1} \approx +0.036
]

from the KNN branch.

---

### Euclidean Branch Remained Necessary

Even when learned:

[
\alpha \approx 0.9
]

the Euclidean branch still improved stability and separation.

---

# 16. Computational Complexity

## Static AVM Only

Training:

[
O(NF)
]

Prediction:

[
O(KF)
]

Memory:

[
O(KF)
]

---

## AVNN + KNN

Prediction:

[
O(NF)
]

per query without indexing.

---

## FAISS-Accelerated AVNN

Approximate search:

[
O(\log N \cdot F)
]

plus:

[
O(kF)
]

for exact re-ranking.

---

# 17. Core Formula Summary

| Component          | Formula                                                            |                                           |   |
| ------------------ | ------------------------------------------------------------------ | ----------------------------------------- | - |
| Normalization      | (\tilde{x}_i = -1 + 2\frac{x_i-\min_i}{\max_i-\min_i})             |                                           |   |
| Angle              | (\theta_i = \arccos(\tilde{x}_i))                                  |                                           |   |
| Transform          | (x'_i = \tanh(0.8\theta_i))                                        |                                           |   |
| Angular Distance   | (\sum_i w_i                                                        | \theta_i(\mathbf{p})-\theta_i(\mathbf{c}) | ) |
| Euclidean Distance | (\sqrt{\sum_i (p'_i-c'_i)^2})                                      |                                           |   |
| Shape Distance     | (|\mathbf{z}(\mathbf{p})-\mathbf{z}(\mathbf{c})|_2)                |                                           |   |
| Blended Distance   | (\beta_a d_a + \beta_e d_e + \beta_s d_s)                          |                                           |   |
| AVM Probability    | (\frac{1/(d_k/\tau+\varepsilon)}{\sum_j 1/(d_j/\tau+\varepsilon)}) |                                           |   |
| KNN Probability    | Weighted inverse-distance vote                                     |                                           |   |
| Final Probability  | (\lambda P_{avm} + (1-\lambda)P_{knn})                             |                                           |   |
| Loss               | Label-smoothed weighted NLL                                        |                                           |   |

---

# 18. Conclusion

AVNN introduces a fundamentally different geometric perspective on classification:

* Per-feature angular geometry
* Hybrid centroid + neighbor inference
* Interpretable branch blending
* Explicit featurewise angular reasoning
* Efficient ANN-compatible embeddings

The framework combines:

* interpretability,
* geometric flexibility,
* differentiability,
* and scalable neighbor retrieval

into a unified classifier architecture.
