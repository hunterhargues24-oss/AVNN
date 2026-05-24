````markdown
# Deep Technical Breakdown of AVNN Mathematics

> Mathematical specification for the Angular Vector Nearest Neighbor (AVNN) family of models, including static, learnable, and FAISS-accelerated variants.

---

# 1. Overview

The Angular Vector Nearest Neighbor (AVNN) framework is a geometric classification architecture built around:

- Axis-separable angular geometry
- Euclidean geometry in transformed angular space
- Optional shape-invariant geometry
- Centroid-based probabilistic inference (AVM branch)
- Weighted k-nearest-neighbor voting (KNN branch)

Unlike cosine similarity, AVNN treats each feature as an independent angular axis and measures per-feature angular deviation directly.

The framework supports both:

- **Static parameter-free models**
- **Learnable differentiable models trained with backpropagation**

---

# 2. Feature Normalization

Let:

```text
x ∈ R^F
````

be a feature vector with `F` features.

Normalization is computed using training-set statistics only.

---

## 2.1 Standard Min-Max Scaling to [0,1]

```text
x_hat_i =
(x_i - min_i)
/
(max_i - min_i + eps)
```

where:

* `min_i` = minimum value of feature `i`
* `max_i` = maximum value of feature `i`
* `eps = 1e-10`

This produces:

```text
x_hat_i ∈ [0,1]
```

---

## 2.2 Recommended Scaling to [-1,1]

AVNN experimentally performs better using full semicircular angular space:

```text
x_tilde_i = 2*x_hat_i - 1
```

giving:

```text
x_tilde_i ∈ [-1,1]
```

This expands the angular domain:

```text
acos(x_tilde_i) ∈ [0, pi]
```

instead of:

```text
[0, pi/2]
```

which empirically improves separability.

---

# 3. Angular Feature Transform

The Euclidean branch operates on a bounded non-linear angular transform:

```text
x'_i =
tanh(
s * acos(x_tilde_i)
)
```

where:

* `s` = scaling factor

Recommended values:

| Input Range | Angular Domain | Recommended s |
| ----------- | -------------- | ------------- |
| [0,1]       | [0, pi/2]      | 1.6           |
| [-1,1]      | [0, pi]        | 0.8           |

This scaling avoids saturation in the `tanh` branch.

---

# 4. Axis-Separable Angular Geometry

AVNN does **not** use standard cosine similarity.

Instead, each feature independently contributes an angular displacement.

For feature `i`:

```text
theta_i(p) = acos(p_tilde_i)
```

For centroid `c`:

```text
theta_i(c) = acos(c_tilde_i)
```

The angular distance is:

```text
d_ang(p,c)
=
sum_{i=1..F}
w_i *
abs(
theta_i(p)
-
theta_i(c)
)
```

where:

```text
w_i >= 0
```

are per-feature weights.

---

## 4.1 Feature Weight Normalization

In learnable variants:

```text
w_i =
F *
exp(a_i)
/
sum_j exp(a_j)
```

where:

* `a_i` are raw learnable logits
* weights sum to `F`

This preserves the scale of a uniform average.

---

# 5. Euclidean Geometry

Euclidean distance is computed in transformed angular space:

```text
d_euc(p,c)
=
|| p' - c' ||_2
```

Expanded:

```text
d_euc
=
sqrt(
sum_{i=1..F}
(p'_i - c'_i)^2
)
```

---

# 6. Shape Geometry

The optional shape branch removes global scale and offset information.

Define:

```text
mu_p
=
(1/F)
*
sum_{i=1..F}
p_tilde_i
```

```text
sigma_p
=
sqrt(
(1/F)
*
sum_{i=1..F}
(p_tilde_i - mu_p)^2
+
eps
)
```

Then:

```text
z_i(p)
=
(p_tilde_i - mu_p)
/
sigma_p
```

Shape distance:

```text
d_shape(p,c)
=
|| z(p) - z(c) ||_2
```

This captures feature-pattern similarity independent of scale.

---

# 7. Blended Distance Function

AVNN combines multiple geometric branches.

For branch weights:

```text
beta_ang,
beta_euc,
beta_shape >= 0
```

with:

```text
beta_ang
+
beta_euc
+
beta_shape
=
1
```

the full blended distance becomes:

```text
d(p,c)
=
beta_ang   * d_ang
+
beta_euc   * d_euc
+
beta_shape * d_shape
```

---

## 7.1 Two-Branch AVNN

The original AVNN formulation uses:

```text
beta_shape = 0
```

and:

```text
beta_ang = alpha
beta_euc = 1 - alpha
```

giving:

```text
d
=
alpha * d_ang
+
(1-alpha) * d_euc
```

---

# 8. AVM Centroid Branch

Each class centroid is computed from training data.

For class `k`:

```text
c_k
=
(1/N_k)
*
sum_{n : y_n = k}
x_n
```

Distances to centroids:

```text
d_k = d(p, c_k)
```

Raw inverse-distance affinity:

```text
r_k =
1 / (d_k + eps)
```

Temperature-scaled affinity:

```text
r_tilde_k =
1 / (d_k / tau + eps)
```

Final AVM probability:

```text
P_avm(k)
=
r_tilde_k
/
sum_j r_tilde_j
```

---

# 9. KNN Branch

AVNN uses hard k-nearest neighbors with inverse-distance weighting.

Let:

```text
N_k(p)
```

be the set of `k` nearest neighbors.

Neighbor weight:

```text
w_i =
1 / (d_i + eps)
```

Normalized:

```text
w_hat_i =
w_i
/
sum_{j ∈ N_k} w_j
```

Class probability:

```text
P_knn(c)
=
sum_{i ∈ N_k}
w_hat_i * 1[y_i = c]
```

---

# 10. Final Prediction

The AVM and KNN branches are blended:

```text
P_final(k)
=
lambda * P_avm(k)
+
(1-lambda) * P_knn(k)
```

where:

```text
lambda ∈ [0,1]
```

is learned via sigmoid:

```text
lambda = sigmoid(a_lambda)
```

Prediction:

```text
y_hat
=
argmax_k P_final(k)
```

---

# 11. Residual Correction Branch

Learnable AVNN variants introduce geometric residual correction.

The model predicts a soft centroid assignment:

```text
p_k =
softmax(
-d_k / tau
)
```

Soft centroid target:

```text
c_soft
=
sum_k p_k * c_k
```

Residual correction:

```text
Delta
=
g * beta * s
⊙
(c_soft - x)
```

where:

* `g = sigmoid(a_g)` = residual gate
* `beta = exp(b)` = residual magnitude
* `s = tanh(u)` = feature scaling vector
* `⊙` = elementwise multiplication

Corrected feature vector:

```text
x_corr
=
x + Delta
```

This replaces earlier teacher-forced centroid correction and removes train/inference mismatch.

---

# 12. Loss Function

The corrected learnable AVNN uses:

```text
NLLLoss
```

on log-probabilities.

---

## 12.1 Negative Log Likelihood

```text
L_NLL
=
-log P(y)
```

---

## 12.2 Label Smoothing

With smoothing parameter `eps_ls`:

```text
L_smooth
=
(1-eps_ls) * L_NLL
+
(eps_ls / K)
*
sum_{k=1..K}
-log P(k)
```

Recommended:

```text
eps_ls = 0.05
```

---

## 12.3 Class Weighting

For class `c`:

```text
v_c
=
N / (K * n_c)
```

where:

* `N` = total samples
* `K` = number of classes
* `n_c` = samples in class `c`

Weights are capped and renormalized.

---

# 13. FAISS Approximation

To accelerate neighbor retrieval, AVNN embeds all branches into a single vector.

Define:

```text
v(p)
=
[
sqrt(beta_ang)   * theta(p),
sqrt(beta_euc)   * p',
sqrt(beta_shape) * z(p)
]
```

where:

```text
theta(p)
=
[
acos(p_tilde_1),
acos(p_tilde_2),
...
acos(p_tilde_F)
]
```

Then:

```text
|| v(p) - v(c) ||_2^2
```

approximates the blended AVNN distance:

```text
beta_ang   * || theta(p)-theta(c) ||_2^2
+
beta_euc   * || p'-c' ||_2^2
+
beta_shape * || z(p)-z(c) ||_2^2
```

This is approximate because the angular branch uses weighted L1 geometry.

After FAISS retrieval, true AVNN distance is recomputed.

---

# 14. Static AVNN Configuration

The best-performing static configuration experimentally observed:

| Parameter            | Value  |
| -------------------- | ------ |
| Normalization        | [-1,1] |
| Transform scale      | 0.8    |
| Angular weight alpha | 0.5    |
| AVM/KNN blend lambda | 0.7    |
| Neighbors k          | 5      |

---

# 15. Empirical Findings

## 15.1 Key Architectural Discoveries

### Full Semicircle Improves Geometry

Using:

```text
[-1,1]
```

instead of:

```text
[0,1]
```

expands angular coverage from:

```text
[0, pi/2]
->
[0, pi]
```

and improves baseline separability.

---

### Saturation Correction Was Critical

Using:

```text
tanh(1.6 * acos(x))
```

on:

```text
[-1,1]
```

caused Euclidean branch saturation.

Reducing scale to:

```text
0.8
```

restored useful Euclidean geometry.

---

### KNN Branch Was Essential

Ablation experiments showed:

```text
Delta F1 ≈ +0.036
```

from the KNN branch.

---

### Euclidean Branch Remained Necessary

Even when learned:

```text
alpha ≈ 0.9
```

the Euclidean branch still improved stability and separation.

---

# 16. Computational Complexity

## Static AVM Only

Training:

```text
O(NF)
```

Prediction:

```text
O(KF)
```

Memory:

```text
O(KF)
```

---

## AVNN + KNN

Prediction:

```text
O(NF)
```

per query without indexing.

---

## FAISS-Accelerated AVNN

Approximate search:

```text
O(log N * F)
```

plus:

```text
O(kF)
```

for exact re-ranking.

---

# 17. Core Formula Summary

| Component          | Formula                                            |   |           |   |     |
| ------------------ | -------------------------------------------------- | - | --------- | - | --- |
| Normalization      | `x_tilde_i = -1 + 2 * ((x_i-min_i)/(max_i-min_i))` |   |           |   |     |
| Angle              | `theta_i = acos(x_tilde_i)`                        |   |           |   |     |
| Transform          | `x'_i = tanh(0.8 * theta_i)`                       |   |           |   |     |
| Angular Distance   | `sum_i w_i * abs(theta_i(p)-theta_i(c))`           |   |           |   |     |
| Euclidean Distance | `sqrt(sum_i (p'_i-c'_i)^2)`                        |   |           |   |     |
| Shape Distance     | `                                                  |   | z(p)-z(c) |   | _2` |
| Blended Distance   | `beta_a*d_a + beta_e*d_e + beta_s*d_s`             |   |           |   |     |
| AVM Probability    | `(1/(d_k/tau+eps)) / sum_j (1/(d_j/tau+eps))`      |   |           |   |     |
| KNN Probability    | Weighted inverse-distance vote                     |   |           |   |     |
| Final Probability  | `lambda*P_avm + (1-lambda)*P_knn`                  |   |           |   |     |
| Loss               | Label-smoothed weighted NLL                        |   |           |   |     |

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

```
```
