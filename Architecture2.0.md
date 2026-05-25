# AutoLambdaAVNN — Architecture Reference v2.0

> A geometric multi-prototype classifier that learns nonlinear angular manifolds over class structure and blends centroid reasoning with FAISS-accelerated neighbourhood inference.

---

# Table of Contents

1. Overview
2. Core Design Philosophy
3. Data Normalisation
4. Geometric Branch System
5. Combined Manifold Representation
6. Prototype Geometry
7. Intra-Class Separation
8. Learned Parameters
9. Training Pipeline
10. Supervised Contrastive Geometry
11. EMA Centroid Stabilisation
12. Mahalanobis Geometry
13. Ordinal Learning
14. Lambda Blending
15. FAISS Inference Engine
16. Loss Functions
17. Branch Interpretation
18. Complexity and Scaling
19. Component Interaction Diagram
20. Parameter Reference
21. Design Tradeoffs
22. Future Directions

---

# 1. Overview

AutoLambdaAVNN is a hybrid geometric learning architecture designed for structured/tabular data.

The system combines:

* Angular geometric embeddings
* Multi-branch nonlinear feature transforms
* Multi-prototype class modelling
* Adaptive Mahalanobis geometry
* FAISS accelerated neighbour search
* Ordinal-aware optimisation
* Supervised contrastive separation
* Adaptive AVM/KNN blending

Unlike tree ensembles that partition feature space with axis-aligned boundaries, AutoLambdaAVNN learns:

* geometric manifolds
* class topology
* local nonlinear regions
* class spread
* intra-class substructure

The model represents each sample as a transformed geometric vector in a learned metric space.

Prediction occurs through two complementary mechanisms:

1. Prototype-based reasoning (AVM)
2. Neighbourhood reasoning (KNN)

Both operate in the SAME learned geometric space.

---

# 2. Core Design Philosophy

The architecture is based on a key assumption:

> Many tabular datasets are not linearly separable in raw Euclidean feature space, but become separable when represented through angular and relational geometry.

The model therefore attempts to:

* convert scalar features into geometric signals
* learn feature curvature
* discover nonlinear class regions
* separate overlapping manifolds
* model subclasses within classes
* preserve local neighbour structure

The architecture behaves similarly to a geometric manifold learner:

Raw Feature Space:

```
Class A: one large cloud
```

Learned Geometric Space:

```
Class A:
    prototype 1
    prototype 2
    prototype 3

Each prototype captures a different nonlinear region.
```

This enables the model to approximate curved decision boundaries without deep neural layers.

---

# 3. Data Normalisation

All features are transformed into [-1, 1]:

genui{"math_block_widget_always_prefetch_v2":{"content":"x_f^{norm}=-1+2\cdot\frac{x_f-\min_f}{\max_f-\min_f}"}}

This provides:

* bounded geometry
* stable angular transforms
* symmetric directional representation
* consistent manifold scaling

The transformation is fit ONLY on training data.

Why [-1,1] instead of [0,1]?

Because angular transforms require signed geometry.

In [0,1]:

* sign information disappears
* opposite directions become indistinguishable

In [-1,1]:

* positive and negative deviation are preserved
* the full angular range [0, π] becomes available

---

# 4. Geometric Branch System

Each feature is transformed through multiple geometric perspectives.

The current architecture contains SIX branches.

---

## 4.1 Linear Angular Branch

Directional geometry:

genui{"math_block_widget_always_prefetch_v2":{"content":"\phi_{lin,f}(x)=\arccos(x_f)"}}

Captures:

* directional position
* signed extremeness
* axis orientation

---

## 4.2 Circular Branch

Magnitude geometry:

genui{"math_block_widget_always_prefetch_v2":{"content":"\phi_{cir,f}(x)=\arccos(|x_f|)"}}

Captures:

* boundary proximity
* extremeness independent of sign
* symmetric manifold structure

---

## 4.3 Boundary Branch

Boundary emphasis transform:

genui{"math_block_widget_always_prefetch_v2":{"content":"\phi_{bnd,f}(x)=\tanh(0.8\cdot\arccos(x_f))"}}

Captures:

* nonlinear edge sensitivity
* compressed long-tail geometry
* asymmetric curvature

This branch tends to specialise in difficult boundary regions.

---

## 4.4 Shape Branch

Within-sample relational geometry:

genui{"math_block_widget_always_prefetch_v2":{"content":"shape_f(x)=\frac{x_f-\bar{x}}{std(x)}"}}

Captures:

* relative feature profile
* proportional structure
* shape independent of absolute scale

This branch often becomes important when classes differ more by feature ratios than raw magnitude.

---

## 4.5 Square Branch

Quadratic curvature transform:

genui{"math_block_widget_always_prefetch_v2":{"content":"sq_f(x)=x_f^2"}}

Captures:

* nonlinear magnitude amplification
* curvature-sensitive structure
* symmetric feature intensity

This branch helps the model represent parabolic separation patterns.

---

## 4.6 Tactical Branch

Adaptive nonlinear transform branch.

The tactical branch acts as a flexible geometric specialist.

Depending on implementation configuration it may include:

* tanh curvature
* adaptive compression
* nonlinear angular reshaping
* feature interaction emphasis

This branch often receives higher weights on difficult datasets.

---

# 5. Combined Manifold Representation

All branch outputs are concatenated into a unified geometric vector.

genui{"math_block_widget_always_prefetch_v2":{"content":"v(x)=[\sqrt{w_1}B_1,\sqrt{w_2}B_2,\dots,\sqrt{w_n}B_n]"}}

Where:

* B_i = branch representation
* w_i = learned branch weight

Feature weighting is also applied:

genui{"math_block_widget_always_prefetch_v2":{"content":"fw=softmax(raw\_feat\_w)\cdot F"}}

This creates a learned geometric manifold where:

* similar classes cluster
* nonlinear regions separate
* local topology is preserved
* neighbour relationships become meaningful

The learned vector space is shared identically between:

* training
* AVM inference
* FAISS KNN inference

No train/inference mismatch exists.

---

# 6. Prototype Geometry

One of the largest upgrades to the architecture is multi-prototype modelling.

Instead of representing a class with ONE centroid:

```
Class A → single mean point
```

The model can represent a class using MULTIPLE prototypes:

```
Class A:
    centroid 1
    centroid 2
    centroid 3
```

Controlled by:

```
n_prototypes
```

Example:

```
n_prototypes=3
```

means:

* each class is subdivided into 3 internal geometric regions
* each region gets its own centroid
* local manifolds become separable

This is extremely important for:

* wine datasets
* multimodal classes
* nonlinear distributions
* overlapping manifolds

---

## 6.1 What Prototypes Actually Learn

The prototypes are NOT arbitrary learned vectors.

They are derived from actual intra-class structure.

Conceptually:

1. samples within a class are clustered
2. each cluster becomes a prototype
3. the prototypes approximate internal submanifolds

Example:

Red Wine Quality:

```
High quality wines may split into:

- high alcohol dry wines
- balanced acidic wines
- fruit-forward wines
```

A single centroid averages these together.

Multiple prototypes preserve the structure.

---

## 6.2 Why Prototypes Improve Nonlinear Geometry

A single centroid assumes:

```
one convex class region
```

Real datasets often look like:

```
multiple disconnected islands
```

Multi-prototype geometry approximates nonlinear manifolds by combining:

* several local convex regions
* neighbour reasoning
* adaptive distances

This effectively creates piecewise nonlinear geometry.

---

# 7. Intra-Class Separation

Another major addition is:

```
intra_sep
```

This controls how strongly prototypes are encouraged to separate from one another.

Conceptually:

Without intra_sep:

```
all prototypes collapse together
```

With intra_sep:

```
prototypes spread across the manifold
```

The architecture therefore learns:

* internal class topology
* subclass geometry
* manifold coverage
* local nonlinear regions

---

## 7.1 Geometric Interpretation

The prototypes become attractors inside the class manifold.

Low intra_sep:

```
small compact cluster
```

High intra_sep:

```
larger distributed manifold coverage
```

This acts similarly to:

* mixture models
* radial basis function centres
* manifold anchors
* local experts

---

## 7.2 Why This Matters

The wine datasets improved substantially after:

* increasing prototypes
* separating prototypes
* enabling supervised contrastive geometry

Because the classes are NOT simple Gaussian clouds.

They contain:

* overlapping chemistry regions
* nonlinear sensory relationships
* multimodal distributions
* ordinal transitions

Multi-prototype separation allows the geometry to capture those hidden regions.

---

# 8. Learned Parameters

The architecture learns multiple parameter groups.

| Parameter              | Purpose                                |
| ---------------------- | -------------------------------------- |
| branch weights         | importance of each geometric transform |
| feature weights        | importance of each feature             |
| tau                    | scoring sharpness                      |
| lambda                 | AVM/KNN blend                          |
| prototype locations    | subclass geometry                      |
| Mahalanobis covariance | class shape                            |

---

## 8.1 Branch Weights

Branch weights are learned through softmax:

genui{"math_block_widget_always_prefetch_v2":{"content":"w=softmax(raw\_w)"}}

This forces:

* all weights positive
* weights sum to 1
* competitive branch learning

---

## 8.2 Feature Weights

Feature weights determine feature importance geometrically.

Higher weight:

* increases manifold distance contribution
* increases neighbour influence
* sharpens separation along that feature

Unlike tree gain metrics, these weights directly alter geometry.

---

## 8.3 Tau

Tau controls softness of inverse distance scoring.

genui{"math_block_widget_always_prefetch_v2":{"content":"p_k\propto\frac{1}{d_k/\tau_k+\epsilon}"}}

Small tau:

* sharper confidence
* harder boundaries

Large tau:

* softer scoring
* smoother manifolds

Per-class tau allows minority classes to remain softer.

---

# 9. Training Pipeline

Training proceeds in several stages.

---

## 9.1 Normalisation

Raw data → bounded geometric space.

---

## 9.2 Branch Construction

All geometric transforms computed.

---

## 9.3 Combined Vector Assembly

All branches concatenated.

---

## 9.4 Prototype Assignment

Each class subdivided into local regions.

---

## 9.5 AVM Training

Distances to prototypes computed.

Inverse-distance probabilities generated.

Cross entropy + auxiliary losses optimised.

---

## 9.6 EMA Stabilisation

Optional EMA smoothing stabilises prototype drift.

---

## 9.7 Post-Training Geometry

After optimisation:

* Mahalanobis covariance estimated
* lambda searched
* FAISS index built
* prototype drift analysed

---

# 10. Supervised Contrastive Geometry

One of the most important upgrades is:

```
supcon=True
```

This introduces supervised contrastive geometry.

The objective:

* pull same-class regions together
* push different-class regions apart
* improve manifold separability

Conceptually:

```
Before:

mixed overlapping geometry

After:

clear separated manifolds
```

This is especially valuable for:

* wine datasets
* noisy datasets
* high overlap datasets
* ordinal transitions

The contrastive signal helps create cleaner neighbour topology.

---

# 11. EMA Centroid Stabilisation

EMA:

```
ema_centroids=True
```

uses exponential moving averages.

Instead of updating prototypes abruptly:

genui{"math_block_widget_always_prefetch_v2":{"content":"\mu_t=\beta\mu_{t-1}+(1-\beta)\hat{\mu}_t"}}

This reduces:

* oscillation
* instability
* noisy prototype jumps

EMA is especially helpful when:

* classes overlap
* prototypes drift rapidly
* mini-batch noise is high

---

# 12. Mahalanobis Geometry

Standard Euclidean geometry assumes:

* all directions equally important
* spherical clusters

Real datasets are not spherical.

Mahalanobis geometry adapts distance to class shape.

genui{"math_block_widget_always_prefetch_v2":{"content":"d_M(x,\mu)=\sqrt{(x-\mu)^T\Sigma^{-1}(x-\mu)}"}}

This allows:

* elongated manifolds
* anisotropic class regions
* adaptive geometry

The architecture currently supports:

* diagonal covariance
* full covariance
* regularised covariance
* RDA blending

---

## 12.1 Full Covariance

Full covariance captures:

* feature interactions
* correlated geometry
* rotated manifolds

This is one reason the wine datasets improved.

Chemical variables are highly correlated.

---

# 13. Ordinal Learning

Wine quality is ordinal.

Predicting:

```
5 instead of 6
```

should NOT be punished the same as:

```
3 instead of 8
```

Ordinal learning addresses this.

The architecture uses Earth Mover's Distance:

genui{"math_block_widget_always_prefetch_v2":{"content":"L_{EMD}=\sum_k|CDF_p[k]-CDF_y[k]|"}}

This encourages:

* smooth ordinal transitions
* adjacent-class tolerance
* rank-aware geometry

This is extremely important for wine quality datasets.

---

# 14. Lambda Blending

Final prediction combines:

* prototype reasoning
* neighbour reasoning

genui{"math_block_widget_always_prefetch_v2":{"content":"p=\lambda p_{AVM}+(1-\lambda)p_{KNN}"}}

AVM:

* global geometric reasoning
* stable prototypes
* class manifold centres

KNN:

* local nonlinear corrections
* neighbourhood adaptation
* fine-grained topology

The blend combines:

* global structure
* local detail

---

# 15. FAISS Inference Engine

The architecture uses FAISS for scalable neighbour retrieval.

Large datasets:

```
200k+
```

remain practical.

The system supports:

* Flat indexes
* IVF indexes
* batched querying
* approximate nearest neighbours

Complexity becomes approximately:

genui{"math_block_widget_always_prefetch_v2":{"content":"O(\log N)"}}

rather than brute-force:

genui{"math_block_widget_always_prefetch_v2":{"content":"O(N)"}}

---

# 16. Loss Functions

Total loss:

genui{"math_block_widget_always_prefetch_v2":{"content":"L=L_{CE}+L_{reg}+L_{EMD}+L_{supcon}+L_{sep}"}}

Components:

| Loss                 | Purpose             |
| -------------------- | ------------------- |
| Cross entropy        | classification      |
| Regularisation       | prevent overfit     |
| EMD                  | ordinal awareness   |
| SupCon               | manifold separation |
| Prototype separation | subclass diversity  |

---

# 17. Branch Interpretation

The learned branch weights reveal dataset geometry.

Example:

High boundary weight:

```
class boundaries are highly nonlinear
```

High shape weight:

```
relative feature structure matters
```

High tactical weight:

```
dataset benefits from nonlinear compression
```

This effectively turns the architecture into a geometric analysis engine.

---

# 18. Complexity and Scaling

Training complexity:

genui{"math_block_widget_always_prefetch_v2":{"content":"O(N\cdot K\cdot P\cdot F)"}}

Where:

* N = samples
* K = classes
* P = prototypes per class
* F = features

Inference complexity with FAISS IVF:

approximately:

genui{"math_block_widget_always_prefetch_v2":{"content":"O(\log N)"}}

This scaling behaviour becomes highly competitive on large datasets.

---

# 19. Component Interaction Diagram

```text
Input Features
      │
      ▼
Normalisation [-1,1]
      │
      ▼
┌─────────────────────────────┐
│ Geometric Branches          │
│                             │
│ Linear Angular              │
│ Circular                    │
│ Boundary                    │
│ Shape                       │
│ Square                      │
│ Tactical                    │
└──────────────┬──────────────┘
               │
               ▼
Combined Geometric Vector
               │
     ┌─────────┴─────────┐
     │                   │
     ▼                   ▼
Prototype Geometry     FAISS Geometry
     │                   │
     ▼                   ▼
AVM Scoring           KNN Scoring
     │                   │
     └─────────┬─────────┘
               ▼
         Lambda Blend
               ▼
          Prediction
```

---

# 20. Parameter Reference

## Core Geometry

| Parameter       | Purpose                        |
| --------------- | ------------------------------ |
| n_prototypes    | subclasses per class           |
| intra_sep       | prototype separation strength  |
| branch weights  | geometric transform importance |
| feature weights | feature geometry importance    |
| tau             | inverse-distance softness      |

---

## Stability

| Parameter       | Purpose                   |
| --------------- | ------------------------- |
| ema_centroids   | centroid smoothing        |
| ortho_reg       | branch decorrelation      |
| feat_weight_reg | feature regularisation    |
| label_smoothing | confidence regularisation |

---

## Geometry

| Parameter           | Purpose                       |
| ------------------- | ----------------------------- |
| mahalanobis         | adaptive covariance geometry  |
| alpha               | covariance shrinkage          |
| full covariance     | correlated manifold modelling |
| diagonal covariance | lightweight geometry          |

---

## Inference

| Parameter | Purpose            |
| --------- | ------------------ |
| lambda    | AVM/KNN blend      |
| k         | neighbour count    |
| nlist     | IVF partitions     |
| nprobe    | IVF search breadth |

---

# 21. Design Tradeoffs

## Why Multi-Prototype Instead of Deep Networks?

Deep networks learn nonlinear manifolds implicitly.

AutoLambdaAVNN approximates them geometrically.

Advantages:

* interpretable geometry
* prototype visibility
* explicit feature importance
* lower parameter count
* strong tabular performance

Tradeoff:

* less expressive than large neural networks
* requires careful geometric engineering

---

## Why Combine AVM and KNN?

AVM provides:

* stable global reasoning
* robust centroid structure

KNN provides:

* local correction
* manifold detail
* boundary adaptation

Together they behave similarly to:

```
global manifold + local residual correction
```

---

## Why Multiple Branches?

Different transforms capture fundamentally different geometry.

Example:

| Branch   | Captures                     |
| -------- | ---------------------------- |
| Linear   | direction                    |
| Circular | extremeness                  |
| Boundary | edge curvature               |
| Shape    | proportional structure       |
| Square   | quadratic intensity          |
| Tactical | adaptive nonlinear structure |

The ensemble of transforms creates richer manifolds than any single transform.

---

# 22. Future Directions

Potential future upgrades:

* dynamic prototype allocation
* hierarchical manifolds
* graph neighbour propagation
* differentiable FAISS approximations
* adaptive branch generation
* learnable nonlinear transforms
* prototype attention routing
* hyperbolic geometry
* manifold curvature learning
* temporal geometric memory

The architecture is evolving toward:

> a fully geometric manifold learning system for structured data.
