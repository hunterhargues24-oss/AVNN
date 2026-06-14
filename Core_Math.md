# The Mathematics of LearningAVNN

A precise, geometry-first account of every mathematical object in the model. The
intended reader is comfortable with linear algebra, quadratic forms, metric
geometry, and a little differential geometry; machine-learning-specific terms are
defined as they appear.

The through-line: LearningAVNN never learns a decision boundary directly. It
learns **a metric and a set of landmarks**, and the boundary is whatever falls out
of "which landmark is nearest" under that metric. Almost every component is best
read as either *a map of the data space*, *a metric on it*, or *a potential field
acting on the landmarks*.

---

## 0. Notation and setup

- $F$ — number of raw features. $K$ — number of classes. $m$ — prototypes per class.
- $x \in \mathbb{R}^F$ — one sample (after normalisation, $x \in [-1,1]^F$).
- $e_i \in \mathbb{R}^F$ — the $i$-th standard basis vector.
- $\Delta^{K-1} = \{p \in \mathbb{R}^K : p_k \ge 0,\ \sum_k p_k = 1\}$ — the probability simplex; its vertices are the one-hot label distributions, its barycentre is the uniform distribution.
- $\mu_{kj} \in \mathbb{R}^F$ — the $j$-th prototype of class $k$ (a learnable landmark).
- $\odot$ — Hadamard (elementwise) product. $\|\cdot\|$ — Euclidean norm unless subscripted.

The model maps $x$ through one or more **feature maps** $\phi:\mathbb{R}^F\to\mathbb{R}^D$, measures distances to landmarks in the image space under a learned metric, converts distances to a point of $\Delta^{K-1}$, and blends several such points.

---

## 1. Normalisation — the working cube

Raw features are min–max scaled to $[-1,1]$:
$$x_i \;\leftarrow\; 2\,\frac{x_i^{\text{raw}} - \min_i}{\max_i - \min_i} - 1.$$

Geometrically the data is placed inside the axis-aligned cube $[-1,1]^F$ centred at
the origin. This matters because the AVM feature map below reads each coordinate as
a **direction cosine** — it is only meaningful when $x_i \in [-1,1]$, i.e. when
$x_i = \cos\theta_i$ for some angle $\theta_i \in [0,\pi]$.

---

## 2. Two diagonal weightings: features and branches

### Feature weights — an anisotropic axis scaling

A learnable vector $r \in \mathbb{R}^F$ defines
$$w \;=\; F\cdot \operatorname{softmax}(r), \qquad \sum_{i=1}^F w_i = F,\quad \text{uniform} \Rightarrow w_i = 1.$$
Each feature enters the maps scaled by $\sqrt{w_i}$. Because the maps are fed into
**squared** distances, a factor $\sqrt{w_i}$ on the coordinate contributes a factor
$w_i$ to the squared distance. So $w$ is a **diagonal metric tensor** $\operatorname{diag}(w)$:
it stretches or compresses each axis. The softmax simplex constraint $\sum w_i = F$
fixes the overall volume scale, so the model can only *redistribute* importance
across axes, not inflate it globally.

### Branch gates — a convex weighting of orthogonal subspaces

Each feature map is a concatenation ("$\oplus$", direct sum) of several **branch**
blocks. Their weights come from **independent** sigmoids, then a normalisation:
$$g_b \;=\; \frac{\sigma(\rho_b)}{\sum_{b'}\sigma(\rho_{b'})}, \qquad g \in \Delta^{B-1}.$$
This is deliberately **not** a softmax. Softmax couples the logits so that raising
one branch necessarily suppresses the others (a zero-sum simplex motion along a
single coordinate). Independent sigmoids let several branches be simultaneously
large; the final normalisation only rescales to the simplex. Each block is scaled
by $\sqrt{g_b}$, so in squared distance the blocks combine as a $g$-convex
combination of per-block squared distances — a weighted direct sum of orthogonal
subspaces.

---

## 3. The AVM feature map — lifting the cube onto circular coordinates

The **AVM** (Angular Vector) space is the *learned* map. It sends each coordinate of
the cube to an angular triple and adds one globally-coupled distance block. Write
$\theta_i = \arccos(x_i) \in [0,\pi]$, so $x_i = \cos\theta_i$.

**tanh–arccos branch** (monotone angular position):
$$\text{tac}_i = \tanh(0.8\,\theta_i)\,\sqrt{w_i} = \tanh\!\big(0.8\,\arccos x_i\big)\sqrt{w_i}.$$
$\arccos$ recovers the angle whose cosine is $x_i$; $\tanh(0.8\,\cdot)$ is a smooth
monotone squash. It is steepest near $x_i=0$ ($\theta_i=\pi/2$) and saturates at the
cube faces $x_i=\pm 1$.

**circular branch** (the sine companion):
$$\text{cir}_i = \sqrt{1 - x_i^2}\,\sqrt{w_i} = |\sin\theta_i|\,\sqrt{w_i}.$$
Since $x_i=\cos\theta_i$, this is $|\sin\theta_i|$ — the height of the point above the
axis on the unit circle. The pair $(x_i,\ \text{cir}_i) = (\cos\theta_i, |\sin\theta_i|)$
places the coordinate on the upper unit semicircle: the AVM map is literally a lift
from the interval $[-1,1]$ to a circular arc, and `tac` is a reparametrised arc
length.

**boundary branch** (distance to cube vertices — the one cross-feature block):
$$\text{bnd}_i = \sqrt{\|x\|^2 - 2x_i + 1} = \|x - e_i\|.$$
This is the Euclidean distance from the whole point $x$ to the axis vertex $e_i$.
Unlike the other two branches it is **not** separable: through the shared $\|x\|^2$
term every output coordinate depends on every input coordinate. This is the AVM
map's only built-in coordinate coupling.

**dual boundary** (`use_dual_boundary`) adds the opposite faces,
$\|x+e_i\| = \sqrt{\|x\|^2 + 2x_i + 1}$, interleaved as
$[\,\|x-e_0\|, \|x+e_0\|, \|x-e_1\|, \dots\,]$, giving distances to all $2F$ axis
vertices $\pm e_i$.

The combined AVM vector (dimension $D = 3F$, or $4F$ with dual boundary) is the
$g$-weighted direct sum:
$$\phi_{\text{AVM}}(x) = \sqrt{g_{\text{tac}}}\,\text{tac} \,\oplus\, \sqrt{g_{\text{bnd}}}\,\text{bnd} \,\oplus\, \sqrt{g_{\text{cir}}}\,\text{cir}.$$

Interpretation: the cube is re-expressed in (cosine, sine, angle) coordinates plus a
vertex-distance field — a representation where "extremeness toward a face" and
"angular position" are explicit axes rather than implicit.

---

## 4. The KNN feature map — a frozen local chart

The **KNN** space is a *separate*, gradient-free map feeding a nearest-neighbour
index. Default blocks (each $\sqrt{g}$-weighted, concatenated):

$$\text{lin}_i = x_i\sqrt{w_i}, \qquad \text{shape}_i = \frac{x_i - \bar x}{s_x}, \qquad \text{quad}_i = x_i^2\sqrt{w_i},$$

where $\bar x, s_x$ are the **per-sample** mean and standard deviation across the $F$
coordinates. `shape` is therefore a within-sample $z$-score: it discards the
sample's overall level and scale and keeps only its *profile* — the relative
geometry of one row, invariant to translation and dilation of that row.

Optional monotone / compositional blocks for imbalanced data:
$$\text{log}_i = \operatorname{sign}(x_i)\log(1+|x_i|), \quad \text{rank}_i = \widehat F_i(x_i)\ \text{(train ECDF)}, \quad \text{clr}_i = \log x_i - \tfrac1F\textstyle\sum_j \log x_j.$$
`log` is a signed log that tames heavy tails (a multiplicative-to-additive
warp); `rank` is the empirical CDF, a strictly monotone, scale-free reparametrisation
onto $[0,1]$; `clr` is the centred-log-ratio used for compositional data (vectors
constrained to a simplex), mapping the simplex to the tangent hyperplane
$\sum_i \text{clr}_i = 0$.

These live in the KNN space precisely **because** it is a flat, uniformly-weighted
Euclidean vote: extra blocks cost nothing but dimensionality. Adding them to the
*AVM* space instead would corrupt its covariance estimate (Section 6) and pollute
the $\|x\|^2$ coupling — a view earns its keep only as a separate, late-fused chart.

---

## 5. Prototype scoring — an inverse-distance potential

In a feature space with image points $\phi(x)$ and landmark images $\phi(\mu_{kj})$,
the class score sums an inverse-distance kernel over that class's prototypes and
normalises across classes:
$$s_k(x) \;=\; \sum_{j=1}^{m}\frac{1}{\,d(\phi(x),\phi(\mu_{kj}))/\tau_k + \varepsilon\,}, \qquad p_k(x) = \frac{s_k(x)}{\sum_{k'} s_{k'}(x)} \in \Delta^{K-1}.$$

Each landmark is a **source** of an inverse-distance potential ($1/d$, the harmonic /
Newtonian kernel in the sense of decaying with distance); the class field is the
superposition of its sources' potentials, and the posterior is the normalised field
strength. As $d\to0$ the score diverges (nearest-landmark dominance); far away it
decays smoothly, so this is a soft, differentiable nearest-prototype rule.

The **per-class temperature** $\tau_k = \exp(\log\tau_k)$ (when `per_class_tau`)
rescales distances *within* a class: a larger $\tau_k$ softens that class's field
(slower decay, longer reach), which lets a minority class claim territory it would
lose under a hard nearest rule. Geometrically $\tau_k$ dilates class $k$'s level
sets.

During **training** $d$ is Euclidean in the AVM image space (so the forward pass is
cheap and differentiable). At **inference** $d$ becomes the Mahalanobis metric of
Section 6.

---

## 6. The Mahalanobis metric and RDA covariance — learned ellipsoids

### The quadratic form

At inference, distance to prototype $kj$ uses a per-prototype symmetric
positive-definite matrix $\Sigma_{kj}$:
$$d_M(x,\mu_{kj})^2 = (x-\mu_{kj})^\top \Sigma_{kj}^{-1}(x-\mu_{kj}).$$
$\Sigma^{-1}$ (the **precision matrix**) is a metric tensor; its level sets
$\{x : d_M = c\}$ are ellipsoids centred at $\mu$. Diagonalising
$\Sigma = Q\Lambda Q^\top$ (orthonormal $Q$, $\Lambda=\operatorname{diag}(\lambda_a)$),
$$d_M^2 = \sum_{a=1}^{D}\frac{\big(q_a^\top (x-\mu)\big)^2}{\lambda_a}.$$
The eigenvectors $q_a$ are the **principal axes** of the class ellipsoid; the
eigenvalues $\lambda_a$ are the squared semi-axis lengths. The change of variables
$y = \Lambda^{-1/2}Q^\top(x-\mu)$ is the **whitening transform**: it carries the
ellipsoidal metric to the ordinary Euclidean one, $d_M^2 = \|y\|^2$. So Mahalanobis
distance is "Euclidean distance after rotating to the class's natural axes and
rescaling each by its spread."

### The precision matrix as an interaction graph

The off-diagonal $(\Sigma^{-1})_{ij}$ is (up to scaling) the negative **partial
correlation** between features $i$ and $j$ given all the others — the edge weight of
the Gaussian graphical model. A nonzero off-diagonal is precisely a learned
**cross-feature interaction**: the quadratic form contains the term
$(\Sigma^{-1})_{ij}(x_i-\mu_i)(x_j-\mu_j)$, a weighted product of two centred
features. This is the model's built-in, principled "feature engineering": it
discovers which feature *combinations* matter (the eigenvectors) and how much (the
inverse eigenvalues), with no hand-specified products. Its reach is exactly
degree-2 (pairwise / ellipsoidal); genuinely higher-order structure must come from
the prototype mixture (Section 7) or a nonlinear map.

### Regularised Discriminant Analysis (RDA) — shrinking the ellipsoid

Two shrinkages keep the estimate well-conditioned when $n_k < D$:
$$\Sigma_{kj} = (1-\alpha)\,S_{kj} + \alpha\,S_{\text{pooled}}, \qquad \Sigma_{kj}^{\text{reg}} = (1-r)\,\Sigma_{kj} + r\,\bar\sigma^2 I,$$
with $\bar\sigma^2 = \operatorname{tr}(\Sigma)/D$ the mean variance. The pooled
estimate is the unbiased within-class scatter
$$S_{\text{pooled}} = \frac{1}{N-K}\sum_{k}\sum_{i\in k}(x_i-\mu_k)(x_i-\mu_k)^\top.$$
- $\alpha$ (`mahal_alpha`) interpolates between **QDA** ($\alpha\to0$: each class
  keeps its own ellipsoid) and **LDA** ($\alpha\to1$: all classes share
  $S_{\text{pooled}}$, the LDA within-class covariance). Geometrically it morphs each
  class's ellipsoid toward a common shared shape.
- $r$ (`mahal_reg`) shrinks every ellipsoid toward an isotropic sphere of radius
  $\bar\sigma$. This guarantees positive-definiteness and lifts near-zero
  eigenvalues off the floor (those tight directions otherwise make $1/\lambda_a$
  explode).

Inversion is by **Cholesky factorisation** with an **escalating-ridge retry** (the
ridge $r$ is multiplied by $10$, up to three times, until $\Sigma$ factorises),
falling back to the Moore–Penrose pseudo-inverse only if all retries fail. Matrices
are held in float64 because the combined-space norms overflow float32 and produce
silent NaNs.

### Per-prototype covariance via soft responsibilities

For $m>1$ each prototype gets its *own* ellipsoid, estimated from the points it
"owns" in a soft sense. With responsibilities (a one-step Gaussian-mixture E-step,
distances taken in the raw normalised space)
$$\text{resp}_{ij} = \operatorname{softmax}_j\!\big(-\|x_i - \mu_{kj}\|\big),$$
the weighted mean and covariance are
$$\mu_j = \frac{\sum_i w_{ij} x_i}{\sum_i w_{ij}}, \qquad S_j = \frac{\sum_i w_{ij}(x_i-\mu_j)(x_i-\mu_j)^\top}{\sum_i w_{ij}},\quad w_{ij}=\text{resp}_{ij}.$$
A weighted estimator has fewer effective samples than rows, so $S_j$ is
bias-corrected by the **effective sample size**
$$\text{ess} = \frac{\big(\sum_i w_{ij}\big)^2}{\sum_i w_{ij}^2}, \qquad S_j \leftarrow S_j\cdot\frac{\text{ess}}{\text{ess}-1}$$
(the weighted analogue of Bessel's $n/(n-1)$). Each sub-cluster thus carries its own
shape — which is the entire point of spreading prototypes with `intra_sep`
(Section 11).

---

## 7. What the decision surface actually looks like

With inverse-Mahalanobis scoring, the region assigned to prototype $kj$ over a rival
$k'j'$ is bounded by the locus of equal distance:
$$(x-\mu_{kj})^\top \Sigma_{kj}^{-1}(x-\mu_{kj}) = (x-\mu_{k'j'})^\top \Sigma_{k'j'}^{-1}(x-\mu_{k'j'}).$$
Expanding, the quadratic terms are $x^\top(\Sigma_{kj}^{-1}-\Sigma_{k'j'}^{-1})x$.

- If the two precisions are **equal** ($\Sigma_{kj}=\Sigma_{k'j'}$, e.g. $\alpha\to1$),
  the quadratic part cancels and the boundary is a **hyperplane** — the LDA regime.
- If they **differ** (QDA regime), the boundary is a general **quadric**
  (ellipsoid, paraboloid, or hyperboloid depending on the signature of the
  difference of precisions).
- With $m>1$, each class owns a *union* of quadric cells, so class regions become
  **piecewise-quadric and possibly non-convex** — the mechanism by which a mixture
  captures multimodal classes (e.g. a quality band spanning two latent sub-modes).

The full Bayes-optimal version of this score is the QDA log-posterior
$$\log p(k\mid x) = \log\pi_k - \tfrac12\log|\Sigma_k| - \tfrac12 d_M(x,\mu_k)^2 + \text{const},$$
whose $\log\pi_k$ and $-\tfrac12\log|\Sigma_k|$ terms the inverse-distance vote omits.
The implemented score keeps the metric geometry (the $d_M$ term) but replaces the
log-posterior bookkeeping with a normalised harmonic vote; the prior term is exposed
separately and optionally in Section 12.

---

## 8. Late fusion — moving inside the probability simplex

Each head outputs a point of $\Delta^{K-1}$. Fusion combines them.

### Legacy two-head blend (AVM + KNN)

$$p = \lambda\, p_{\text{AVM}} + (1-\lambda)\,p_{\text{KNN}}.$$
For fixed $\lambda$ this is a **chord** of the simplex between the two heads'
predictions. Three regimes set $\lambda$:

- **Scalar**: grid-searched on $[\lambda_{\text{floor}},1]$ to maximise the
  validation metric.
- **Confidence gate**: per sample, using normalised Shannon entropy
  $$H(p) = -\frac{1}{\log K}\sum_k p_k\log p_k \in [0,1], \qquad c = 1 - H,$$
  $$\lambda = \operatorname{clip}\!\Big(\frac{c_{\text{AVM}}}{c_{\text{AVM}}+c_{\text{KNN}}},\ \lambda_{\text{unc}},\ \lambda_{\text{conf}}\Big).$$
  Here $H=0$ at a simplex vertex (a confident one-hot prediction) and $H=1$ at the
  barycentre (maximal uncertainty), so $c$ measures proximity to a vertex; the head
  closer to a vertex gets more weight.
- **Entropy gate**: $\lambda(x) = \lambda_{\text{conf}} - (\lambda_{\text{conf}}-\lambda_{\text{unc}})\,H(p_{\text{AVM}})$ — lean on AVM when it is confident, defer to KNN as it grows uncertain.

A per-sample **disagreement** diagnostic uses the geometry of the two distributions:
$$\text{overlap} = \sum_k \min(p^{\text{AVM}}_k, p^{\text{KNN}}_k), \qquad \text{disagree} = c_{\text{AVM}}\,c_{\text{KNN}}\,(1-\text{overlap}),$$
i.e. both heads confident *and* their distributions nearly disjoint.

### N-head fusion (any larger head set)

For heads $h$ with predictions $p_h$, the fused point is a convex combination
$\sum_h \omega_h p_h$, $\omega \in \Delta^{H-1}$ — a point in the convex hull of the
heads' predictions. Two candidate weightings:

- **Confidence weights**: $\omega_h \propto c_h = 1 - H(p_h)$, normalised — a
  per-sample reweighting toward whichever head is nearest a vertex.
- **Static weights**: coordinate-ascent on the validation metric, then
  **regularised** to resist overfitting a small/imbalanced validation set:
  $$\omega \leftarrow (1-s)\,\omega + s\,\mathbf{u}, \qquad \omega_h \ge \text{floor},$$
  with $\mathbf{u}$ the uniform (barycentric) weight, $s=0.5$ the shrink, and a floor
  so no head is ever zeroed (no motion onto a face of the head-simplex). Selection
  prefers $\text{uniform} \prec \text{confidence} \prec \text{static}$, only moving
  up when validation improves by a margin. Geometrically: stay near the barycentre
  of the head-simplex unless the data clearly justifies a corner.

---

## 9. The Fisher head — the most-separating linear subspace

Linear Discriminant Analysis solves the generalised eigenproblem on the
between- and within-class scatter matrices
$$S_B = \sum_k n_k (\mu_k - \mu)(\mu_k-\mu)^\top, \qquad S_W = \sum_k\sum_{i\in k}(x_i-\mu_k)(x_i-\mu_k)^\top,$$
$$S_B\,v = \lambda\,S_W\,v \quad\Longleftrightarrow\quad \text{maximise the Rayleigh quotient } \frac{v^\top S_B v}{v^\top S_W v}.$$
The leading eigenvectors are the directions that maximise between-class spread per
unit within-class spread — the most class-separating **linear combinations** of
features, spanning a subspace of dimension $\le K-1$. The implementation uses the
`eigen` solver with automatic (Ledoit–Wolf) shrinkage of $S_W$, then scores by the
Gaussian posterior with shared covariance. Where the per-prototype Mahalanobis of
Section 6 is the *local* cross-feature engine, the Fisher head is its *global,
discriminative* counterpart — which is also why it underperforms on multimodal,
severely imbalanced targets, where a single global linear projection is the wrong
shape.

---

## 10. The geodesic head — unfolding a curved manifold (Isomap)

When data lies on a curved low-dimensional manifold embedded in $\mathbb{R}^F$,
Euclidean distance "cuts through" the ambient space and misrepresents intrinsic
proximity. Isomap repairs this:

1. Build a $k$-nearest-neighbour graph with Euclidean edge weights.
2. Approximate the **geodesic** (intrinsic) distance between every pair as the graph
   **shortest path** — distance along the manifold rather than through the ambient
   space.
3. Apply **classical multidimensional scaling** to the geodesic distance matrix:
   double-centre $-\tfrac12 D^{(2)}$ and take the top eigenvectors, giving a
   low-dimensional Euclidean embedding that preserves geodesic distances as well as
   possible.

The head then scores by inverse-distance to per-class centroids **in the unfolded
embedding**. Geometrically it flattens the manifold into a Euclidean chart and does
nearest-centroid there. It is fit on a subsample (`geodesic_max_fit`) for cost and
is the least-validated head — off by default.

---

## 11. The training objective — a sum of geometric potentials

The model trains the AVM space (and prototype positions) by minimising a composite
loss. Each term has a clean geometric reading.

### Weighted negative log-likelihood

$$\mathcal{L}_{\text{NLL}} = -\sum_i w_{y_i}\log p_{y_i}(x_i), \qquad w_k = \min\!\Big(\frac{N}{K\,n_k},\ \text{cap}\Big).$$
Cross-entropy equals $\mathrm{KL}(\text{onehot}\,\|\,p)$ up to a constant, so each
sample is pulled toward its true simplex vertex; the **balanced, capped** class
weights $w_k$ strengthen that pull for rare classes (the cap prevents a tiny class
from dominating the gradient).

### Label smoothing — a target moved off the vertex

The target distribution is interpolated toward the barycentre,
$q = (1-\alpha)\,\text{onehot} + \tfrac{\alpha}{K}\mathbf{1}$, giving
$$\mathcal{L}_{\text{CE}} = (1-\alpha)\,\mathcal{L}_{\text{NLL}} + \alpha\cdot\operatorname{mean}_k\big(-\log p_k\big).$$
The model is asked to aim not at the corner but at a point $\alpha$ of the way to the
centre — a margin against overconfident corners.

### Ordinal EMD — 1-Wasserstein on the label line

For ordered labels, with $C_p(k)=\sum_{j\le k}p_j$ the predicted CDF and
$C_y$ the true step CDF,
$$\mathcal{L}_{\text{EMD}} = \frac{1}{K-1}\sum_{k=1}^{K-1}\big|C_p(k) - C_y(k)\big|.$$
This is the **Earth Mover's / 1-Wasserstein distance** between the predicted and true
distributions on the 1-D ordered label axis: the work to transport probability mass
from where it is to where it should be. Unlike NLL it charges *proportionally to how
far off* an ordinal mistake is.

### Supervised contrastive loss — clustering on the unit sphere

Embeddings are $L_2$-normalised onto the sphere $S^{D-1}$, and similarity is cosine,
$\text{sim}(z_i,z_j)=z_i^\top z_j$. For anchor $i$ with same-class set $P(i)$,
$$\mathcal{L}_{\text{SupCon}} = \sum_i \frac{-1}{|P(i)|}\sum_{p\in P(i)} \log\frac{\exp(\text{sim}(z_i,z_p)/T)}{\sum_{a\ne i}\exp(\text{sim}(z_i,z_a)/T)}.$$
On the hypersphere this contracts same-class points and expands different-class
points by **angular** distance; temperature $T$ sets the sharpness. It builds tight
minority clusters that NLL alone leaves diffuse.

### Branch orthogonality — pushing the Gram matrix to the identity

Per-branch chunks are $L_2$-normalised; their mean batch Gram matrix
$G = \operatorname{mean}_i\, \hat\phi_i \hat\phi_i^\top$ is penalised toward identity:
$$\mathcal{L}_{\text{ortho}} = \rho\,\|G - I\|_F^2.$$
This drives the branch representations toward mutual orthonormality — decorrelated
subspaces, so branches carry complementary rather than redundant geometry.

### Prototype potential field

The landmarks feel three forces:
$$\underbrace{\beta\,\|\,C - C_0\,\|^2}_{\text{anchor (harmonic well at the init means)}} \;-\; \underbrace{\gamma\,\overline{\,\|\mu_{k}-\mu_{k'}\|\,}_{k\ne k'}}_{\text{inter-class repulsion}} \;-\; \underbrace{\eta\,\overline{\,\|\mu_{kj}-\mu_{kj'}\|\,}_{j\ne j'}}_{\text{intra-class repulsion}}.$$
The anchor is a quadratic spring pinning each prototype near its class mean; the two
separation terms are repulsive potentials (linear in distance, hence constant-force)
pushing different classes apart and, for $m>1$, a class's own prototypes apart so
they spread to cover sub-modes. (These separation terms must remain attached to the
autograd graph; if detached they exert no force — a bug that once silently disabled
them.)

A final spring $\mathcal{L}_{\text{fw}} = \kappa\,\operatorname{mean}((w-1)^2)$ pulls
the diagonal feature metric back toward isotropy.

---

## 12. Decision-stage prior gravity

An optional, post-fusion reweight by the empirical class prior $\pi$ raised to a
temperature $\alpha$ (`prior_temp`):
$$p'(k\mid x) \;\propto\; p(k\mid x)\,\pi_k^{\alpha} \quad\Longleftrightarrow\quad \log p' = \log p + \alpha\log\pi + \text{const}.$$
In **additive-log-ratio coordinates** on the simplex (the natural chart in which the
simplex is a vector space), this is simply a **translation by $\alpha\log\pi$** —
sliding every posterior the same amount along the prior direction. $\alpha=0$ is the
identity; $\alpha=1$ restores the Bayes prior term dropped from the inverse-distance
score (Section 7).

Why it is empirically dominated on severe imbalance: the fused $p$ is a
low-dynamic-range distribution (a normalised inverse-distance vote rarely approaches
a vertex), while $\log\pi$ spans $\sim 3$ nats between an $80\%$ and a $4\%$ class.
The translation therefore overwhelms the likelihood at any non-trivial $\alpha$,
collapsing decisions toward the majority vertex (positive $\alpha$) or the minority
(negative $\alpha$). The frontier-optimal value is $\alpha=0$; the correct lever for
the accuracy/macro-F1 trade-off is the per-class temperature $\tau_k$ (Section 5),
which acts where the probabilities are *formed* rather than translating them after
the fact.

---

## 13. The validation metric — one confusion matrix, three views

Early stopping and fusion selection use a single composite computed from the
confusion matrix $C$ (so accuracy, macro-F1, and weighted-F1 share one pass). With
per-class precision/recall/F1 from $C$,
$$\mathcal{V} = (1-b)\cdot\tfrac12\big(\text{acc} + \text{F1}_{\text{weighted}}\big) + b\cdot\text{F1}_{\text{macro}}, \qquad b = \texttt{val\_macro\_bias}.$$
$b$ slides the selection criterion from prevalence-weighted performance ($b=0$,
accuracy-like) to class-balanced performance ($b=1$, macro-F1) — a single scalar
controlling where on the accuracy/rare-class trade-off the model is tuned.

---

## 14. Optimisation

Parameters are updated with Adam under a `CosineAnnealingWarmRestarts` schedule
($T_0=30$, $T_{\text{mult}}=2$): the learning rate follows a half-cosine decay to
near zero, then restarts, with each cycle twice as long as the last. Gradients are
clipped to unit norm (project the gradient back into the unit ball when it exceeds
it) to bound the step in the curved AVM geometry. Early stopping validates after a
warmup on the metric $\mathcal{V}$ above.

---

## 15. The cross-feature picture, in one paragraph

Per-feature importance is a **diagonal** metric — it can only stretch the coordinate
axes. Cross-feature importance requires the **off-diagonal** structure of a full
metric: the precision matrix $\Sigma^{-1}$, whose eigenvectors are the learned
feature *combinations* and whose inverse eigenvalues are their importances (the
steep directions of the metric). The prototype mixture extends this from one
ellipsoid to a piecewise-quadric, capturing combinations that matter only locally;
the Fisher head supplies the global discriminative combinations. The natural next
step — a single learned low-rank projection $L$ scored in $Lx$-space — would
generalise the diagonal feature weights of Section 2 to feature *combinations*
directly, learned from the objective and stable under imbalance, with the rows of
$L$ readable as the engineered features the model chose.
