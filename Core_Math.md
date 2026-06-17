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
- $\Delta^{K-1} = \{p \in \mathbb{R}^K : p_k \ge 0,\ \sum_k p_k = 1\}$ — the probability simplex; its vertices are the one-hot label distributions, its barycentre the uniform distribution.
- $\mu_{kj} \in \mathbb{R}^F$ — the $j$-th prototype of class $k$ (a learnable landmark).
- $\odot$ — Hadamard product. $\|\cdot\|$ — Euclidean norm unless subscripted.

The model maps $x$ through one or more **feature maps** $\phi:\mathbb{R}^F\to\mathbb{R}^D$, measures distances to landmarks in the image space under a learned metric, converts distances to a point of $\Delta^{K-1}$, and blends several such points.

---

## 1. Normalisation — the working cube

Raw features are min–max scaled to $[-1,1]$:
$$x_i \;\leftarrow\; 2\,\frac{x_i^{\text{raw}} - \min_i}{\max_i - \min_i} - 1.$$

Geometrically the data is placed inside the axis-aligned cube $[-1,1]^F$ centred at
the origin. This matters because the AVM feature map reads each coordinate as a
**direction cosine** — meaningful only when $x_i \in [-1,1]$, i.e. when
$x_i = \cos\theta_i$ for some $\theta_i \in [0,\pi]$.

---

## 2. Two diagonal weightings: features and branches

### Feature weights — an anisotropic axis scaling

A learnable vector $r \in \mathbb{R}^F$ defines
$$w \;=\; F\cdot \operatorname{softmax}(r), \qquad \sum_{i=1}^F w_i = F,\quad \text{uniform} \Rightarrow w_i = 1.$$
Each feature enters scaled by $\sqrt{w_i}$. Fed into **squared** distances, that
$\sqrt{w_i}$ contributes a factor $w_i$, so $w$ is a **diagonal metric tensor**
$\operatorname{diag}(w)$ stretching each axis. The simplex constraint $\sum w_i=F$
fixes the volume, so the model only *redistributes* importance across axes.

### Branch gates — a convex weighting of orthogonal subspaces

Each feature map is a direct sum ("$\oplus$") of **branch** blocks, weighted by
**independent** sigmoids then normalised:
$$g_b \;=\; \frac{\sigma(\rho_b)}{\sum_{b'}\sigma(\rho_{b'})}, \qquad g \in \Delta^{B-1}.$$
Deliberately **not** softmax: softmax couples logits so raising one branch
suppresses the rest; independent sigmoids let several branches be large at once.
Each block is scaled by $\sqrt{g_b}$, so in squared distance the blocks combine as a
$g$-convex combination of per-block squared distances.

---

## 3. The AVM feature map — lifting the cube onto circular coordinates

The **AVM** (Angular Vector) space is the *learned* map. Write
$\theta_i = \arccos(x_i)$.

- **tanh–arccos** (monotone angular position): $\text{tac}_i = \tanh(0.8\,\theta_i)\sqrt{w_i}$ — steepest at $x_i=0$, saturating at the faces $x_i=\pm1$.
- **circular** (the sine companion): $\text{cir}_i = \sqrt{1-x_i^2}\,\sqrt{w_i} = |\sin\theta_i|\sqrt{w_i}$. The pair $(\cos\theta_i,|\sin\theta_i|)$ lifts the interval to the unit semicircle.
- **boundary** (vertex distance — the one cross-feature block): $\text{bnd}_i = \sqrt{\|x\|^2 - 2x_i + 1} = \|x - e_i\|$. Through the shared $\|x\|^2$ every output coordinate depends on every input — the AVM map's only built-in coupling. It is **not** feature-weighted (the $\|x\|^2$ coupling makes per-feature weighting ill-defined).

`use_dual_boundary` adds $\|x+e_i\| = \sqrt{\|x\|^2+2x_i+1}$ interleaved, giving all $2F$ vertices $\pm e_i$.

Combined AVM vector ($D=3F$, or $4F$ with dual boundary):
$$\phi_{\text{AVM}}(x) = \sqrt{g_{\text{tac}}}\,\text{tac} \,\oplus\, \sqrt{g_{\text{bnd}}}\,\text{bnd} \,\oplus\, \sqrt{g_{\text{cir}}}\,\text{cir}.$$

---

## 4. The KNN feature map — a frozen local chart

A *separate*, gradient-free map feeding a nearest-neighbour index. Default blocks:
$$\text{lin}_i = x_i\sqrt{w_i}, \qquad \text{shape}_i = \frac{x_i - \bar x}{s_x}, \qquad \text{quad}_i = x_i^2\sqrt{w_i},$$
with $\bar x, s_x$ the **per-sample** mean/sd across coordinates, so `shape` is a
within-sample $z$-score — pure profile, invariant to a row's level and scale.

Optional blocks for imbalanced/compositional data:
$$\text{log}_i = \operatorname{sign}(x_i)\log(1+|x_i|), \quad \text{rank}_i = \widehat F_i(x_i)\ \text{(train ECDF)}, \quad \text{clr}_i = \log x_i - \tfrac1F\textstyle\sum_j \log x_j.$$

**interaction** (`'interaction'` in `knn_branches`) — a probe for nonlinear
cross-feature structure. Take all $\binom{F}{2}$ off-diagonal products $x_ax_b$,
project them through a fixed seeded Gaussian $P$ to $\texttt{interaction\_dim}$
coordinates, and $z$-score on the training set:
$$\text{intx}(x) = \frac{(\,\text{vech}_{a<b}(x_ax_b)\,)\,P - \mu_{\text{tr}}}{\sigma_{\text{tr}}}.$$
A frozen random-feature (kitchen-sinks) approximation of the degree-2 polynomial
map. It answers *whether* interactions carry signal, not *which* pairs.

These live in the KNN space **because** it is a flat, uniform Euclidean vote: extra
blocks cost only dimensionality. Adding them to the *AVM* space corrupts its
covariance estimate (§6) and pollutes the $\|x\|^2$ coupling — a view earns its keep
as a separate, late-fused chart.

---

## 5. Prototype scoring — an inverse-distance potential

In a space with image points $\phi(x)$ and landmark images $\phi(\mu_{kj})$, the
class score sums an inverse-distance kernel over a class's prototypes:
$$s_k(x) \;=\; \sum_{j=1}^{m}\frac{1}{\,d(\phi(x),\phi(\mu_{kj}))/\tau_k + \varepsilon\,}, \qquad p_k(x) = \frac{s_k(x)}{\sum_{k'} s_{k'}(x)}.$$
Each landmark is a **source** of a $1/d$ potential; the posterior is the normalised
field strength — a soft, differentiable nearest-prototype rule. The **per-class
temperature** $\tau_k = e^{\log\tau_k}$ dilates class $k$'s level sets, letting a
minority class claim territory a hard nearest rule would deny it.

**Two-stage metric.** During **training** $d$ is Euclidean in the AVM image space
(cheap, differentiable), so the learned prototypes are Euclidean-optimal and
$\tau_k$ is a *training-only* knob. At **inference** $d$ becomes the Mahalanobis
metric of §6, and the score becomes the QDA posterior of §7.

---

## 6. The Mahalanobis metric and RDA covariance — learned ellipsoids

### The quadratic form

$$d_M(x,\mu_{kj})^2 = (x-\mu_{kj})^\top \Sigma_{kj}^{-1}(x-\mu_{kj}).$$
$\Sigma^{-1}$ (the **precision matrix**) is a metric tensor with ellipsoidal level
sets. With $\Sigma = Q\Lambda Q^\top$, the whitening $y=\Lambda^{-1/2}Q^\top(x-\mu)$
gives $d_M^2=\|y\|^2$: Mahalanobis distance is Euclidean distance after rotating to
the class's natural axes and rescaling each by its spread.

### Standardisation before estimation

The combined AVM vector stacks branches on very different scales (the boundary
block grows like $\sqrt F$; `tac`/`cir` sit in $[-1,1]$). Mahalanobis distance is
invariant to a fixed diagonal rescale — but the ridge term and the RDA pooling
below are **not**. So each combined column is $z$-scored (a stored
$\texttt{avm\_scale}$) before $\Sigma$ is estimated, and the same scaling is applied
to queries and prototype centres at scoring time. Without it the metric is
dominated by the boundary block.

### The precision matrix as an interaction graph

The off-diagonal $(\Sigma^{-1})_{ij}$ is, up to scale, the negative **partial
correlation** between features $i,j$ — the edge of a Gaussian graphical model. A
nonzero off-diagonal is a learned **cross-feature interaction**: the form contains
$(\Sigma^{-1})_{ij}(x_i-\mu_i)(x_j-\mu_j)$, a weighted product of two centred
features. This is the model's principled, hand-free degree-2 feature engineering —
and the reason the old explicit "triangle"/"bilinear" pairwise scorers were removed
as redundant. Its reach is exactly pairwise; higher order comes from the prototype
mixture (§7) or a nonlinear map (§15).

### RDA shrinkage

$$\Sigma_{kj} = (1-\alpha)\,S_{kj} + \alpha\,S_{\text{pooled}}, \qquad \Sigma_{kj}^{\text{reg}} = (1-r)\,\Sigma_{kj} + r\,\bar\sigma^2 I.$$
$\alpha$ (`mahal_alpha`) morphs each ellipsoid between **QDA** ($\alpha\to0$, own
shape) and **LDA** ($\alpha\to1$, shared $S_{\text{pooled}}$). $r$ (`mahal_reg`)
shrinks toward an isotropic sphere, guaranteeing positive-definiteness and lifting
tiny eigenvalues off the floor (where $1/\lambda_a$ would explode).

Inversion is **Cholesky** with an **escalating-ridge retry** ($r\times10$, up to 3×)
before a pseudo-inverse fallback; the **log-determinant** $\log|\Sigma_{kj}| =
2\sum\log\operatorname{diag}(L)$ is read straight off the Cholesky factor $L$ for the
QDA volume term (§7). float64 throughout (combined-space norms overflow float32).

### Per-prototype covariance via soft responsibilities

For $m>1$ each prototype gets its own ellipsoid from the points it softly owns,
$\text{resp}_{ij}=\operatorname{softmax}_j(-\|x_i-\mu_{kj}\|)$, with weighted
mean/covariance and an **effective-sample-size** (weighted Bessel) correction
$\text{ess}=(\sum w)^2/\sum w^2$, $S_j \leftarrow S_j\cdot\text{ess}/(\text{ess}-1)$.
Each sub-cluster carries its own shape — the point of spreading prototypes with
`intra_sep` (§11).

---

## 7. The decision surface and the QDA posterior

With Mahalanobis scoring, the region of prototype $kj$ over rival $k'j'$ is bounded
by equal distance; the quadratic part is $x^\top(\Sigma_{kj}^{-1}-\Sigma_{k'j'}^{-1})x$.

- Equal precisions ($\alpha\to1$): quadratic cancels → **hyperplane** (LDA).
- Differing precisions (QDA): a general **quadric**.
- $m>1$: a *union* of quadric cells per class → **piecewise-quadric, possibly
  non-convex** regions, capturing multimodal classes.

**The implemented score is the QDA posterior** (`avm_score='qda'`, the default).
Earlier versions used only the harmonic inverse-distance vote, dropping the prior
and volume terms; those terms are now first-class, as an equal-weight mixture over a
class's prototypes:
$$\log p(k\mid x) = \texttt{prior\_weight}\cdot\log\pi_k + \operatorname*{logsumexp}_{j}\Big[-\tfrac12\big(\texttt{logdet\_weight}\cdot\log|\Sigma_{kj}| + d_M(x,\mu_{kj})^2\big)\Big],$$
softmaxed over $k$ (the $-\tfrac12 D\log 2\pi$ and $-\log m$ constants cancel).

Two dials temper the Bayes form against small-sample fragility:
- `logdet_weight` scales the volume term $-\tfrac12\log|\Sigma_k|$. At high $D$/low
  $n$ the log-determinant is dominated by ridge-floored noise directions and injects
  fold variance; setting it $<1$ keeps the metric geometry without the volume
  bookkeeping.
- `prior_weight` scales $\log\pi_k$. On severe imbalance the prior pulls toward the
  majority vertex and costs minority recall / macro-F1; setting it $0$ drops the
  prior entirely.

The alternative `avm_score='inverse_distance'` restores the flat harmonic vote
$s_k=\sum_j 1/(d_M/\tau_k+\varepsilon)$ — heavy-tailed and better-calibrated for the
entropy gates of §8; there `logdet_weight` has no analogue and `prior_weight` enters
multiplicatively as $\pi_k^{\,\texttt{prior\_weight}}$. The pre-QDA behaviour is
exactly `inverse_distance` with `prior_weight=0`.

---

## 8. Late fusion — moving inside the probability simplex

Each head outputs a point of $\Delta^{K-1}$; fusion combines them.

### Legacy two-head blend (AVM + KNN)

$$p = \lambda\, p_{\text{AVM}} + (1-\lambda)\,p_{\text{KNN}}.$$
A **chord** of the simplex. Three regimes for $\lambda$:
- **Scalar**: grid-searched on $[\lambda_{\text{floor}},1]$ for the validation metric.
- **Confidence gate**: with normalised entropy $H(p)=-\tfrac1{\log K}\sum p_k\log p_k$ and $c=1-H$, $\lambda=\operatorname{clip}\!\big(\tfrac{c_{\text{AVM}}}{c_{\text{AVM}}+c_{\text{KNN}}},\lambda_{\text{unc}},\lambda_{\text{conf}}\big)$ — the head nearer a vertex leads.
- **Entropy gate**: $\lambda(x)=\lambda_{\text{conf}}-(\lambda_{\text{conf}}-\lambda_{\text{unc}})H(p_{\text{AVM}})$.

Disagreement diagnostic: $\text{overlap}=\sum_k\min(p^{\text{AVM}}_k,p^{\text{KNN}}_k)$, $\text{disagree}=c_{\text{AVM}}c_{\text{KNN}}(1-\text{overlap})$.

### N-head fusion

Convex combination $\sum_h\omega_h p_h$ in the hull of the heads' predictions.
**Confidence** weights $\omega_h\propto 1-H(p_h)$, or **static** weights from
coordinate-ascent on validation then **regularised**: $\omega\leftarrow(1-s)\omega+s\mathbf{u}$
with a per-head floor (no head zeroed). Selection prefers
uniform $\prec$ confidence $\prec$ static, moving up only past a margin — stay near
the barycentre unless the data justify a corner.

---

## 9. The Fisher head — the most-separating linear subspace

LDA solves $S_B v = \lambda S_W v$ (maximise $v^\top S_B v / v^\top S_W v$): the
directions of greatest between-class spread per unit within-class spread, a subspace
of dim $\le K-1$. `eigen` solver, Ledoit–Wolf shrinkage of $S_W$, Gaussian posterior
with shared covariance. Where §6's Mahalanobis is the *local* cross-feature engine,
Fisher is its *global, discriminative* counterpart — which is why it underperforms
on multimodal, severely imbalanced targets where one global linear projection is the
wrong shape.

---

## 10. The geodesic head — unfolding a curved manifold (Isomap)

Build a $k$-NN graph, approximate **geodesic** distances as graph shortest paths,
apply classical MDS to embed in low dimensions, score by inverse-distance to
per-class centroids in the unfolded chart. Fit on a subsample (`geodesic_max_fit`);
least-validated, off by default.

---

## 11. The training objective — a sum of geometric potentials

### Weighted NLL
$$\mathcal{L}_{\text{NLL}} = -\sum_i w_{y_i}\log p_{y_i}(x_i), \qquad w_k = \min\!\Big(\tfrac{N}{K\,n_k},\ \text{cap}\Big),$$
cross-entropy = $\mathrm{KL}(\text{onehot}\,\|\,p)$ up to a constant: each sample is
pulled to its true vertex, harder for rare classes (capped to stop a tiny class
dominating).

### Label smoothing
Target moved toward the barycentre, $q=(1-\alpha)\text{onehot}+\tfrac{\alpha}{K}\mathbf1$:
$$\mathcal{L}_{\text{CE}} = (1-\alpha)\mathcal{L}_{\text{NLL}} + \alpha\cdot\operatorname{mean}_k(-\log p_k).$$

### Ordinal EMD
$$\mathcal{L}_{\text{EMD}} = \tfrac1{K-1}\textstyle\sum_{k}|C_p(k)-C_y(k)|,$$
the 1-Wasserstein distance between predicted and true CDFs on the ordered label
axis — charges *proportionally* to how far off an ordinal mistake is.

### Supervised contrastive
$L_2$-normalise to the sphere, cosine similarity; for anchor $i$ with same-class set $P(i)$,
$$\mathcal{L}_{\text{SupCon}} = \sum_i \tfrac{-1}{|P(i)|}\sum_{p\in P(i)} \log\frac{\exp(z_i^\top z_p/T)}{\sum_{a\ne i}\exp(z_i^\top z_a/T)},$$
contracting same-class and expanding different-class points by angular distance —
tight minority clusters NLL alone leaves diffuse.

### Branch orthogonality
Per-branch chunks $L_2$-normalised; mean Gram $G$ penalised toward identity,
$\mathcal{L}_{\text{ortho}}=\rho\|G-I\|_F^2$, decorrelating the three AVM branches
(under dual boundary the $2F$ block is collapsed to $F$ first so the grouping stays
over the true three semantic branches).

### Prototype potential field
$$\underbrace{\beta\|C-C_0\|^2}_{\text{anchor spring}} - \underbrace{\gamma\,\overline{\|\mu_k-\mu_{k'}\|}}_{\text{inter-class repulsion}} - \underbrace{\eta\,\overline{\|\mu_{kj}-\mu_{kj'}\|}}_{\text{intra-class repulsion}},$$
plus a feature-weight spring $\kappa\,\operatorname{mean}((w-1)^2)$ toward isotropy.
The separation terms must stay attached to the autograd graph — detaching them
exerts no force (a fixed bug).

---

## 12. Two prior mechanisms

There are now two places the empirical class prior $\pi$ can enter:

1. **In the discriminant** — `prior_weight` multiplies $\log\pi_k$ inside the QDA
   score (§7). This is the principled location: the prior enters where the posterior
   is *formed*. Default 1 (full Bayes prior); 0 drops it.
2. **Post-fusion gravity** — `prior_temp` reweights the *fused* posterior,
   $p'(k\mid x)\propto p(k\mid x)\,\pi_k^{\alpha}$, i.e. $\log p' = \log p + \alpha\log\pi$.
   In additive-log-ratio coordinates this is a **translation** by $\alpha\log\pi$.

The gravity term is empirically dominated on severe imbalance: the fused $p$ is
low-dynamic-range (a vote rarely nears a vertex) while $\log\pi$ spans $\sim3$ nats
between an $80\%$ and a $4\%$ class, so any non-trivial $\alpha$ collapses decisions
toward a vertex. Keep `prior_temp=0` and steer the accuracy/macro-F1 trade-off with
the per-class $\tau_k$ (§5), `val_macro_bias` (§13), and `weight_cap` — levers that
act where probabilities are formed, not translations after the fact.

---

## 13. The validation metric — one confusion matrix, three views

Early stopping and fusion selection use a single composite from the confusion matrix
(accuracy, macro-F1, weighted-F1 in one pass):
$$\mathcal{V} = (1-b)\cdot\tfrac12(\text{acc}+\text{F1}_{\text{w}}) + b\cdot\text{F1}_{\text{macro}}, \qquad b=\texttt{val\_macro\_bias}.$$
$b$ slides selection from prevalence-weighted ($b=0$) to class-balanced ($b=1$). The
same machinery backs `report()`, which exposes the per-class precision/recall/F1 and
confusion directly.

---

## 14. Optimisation

Adam under `CosineAnnealingWarmRestarts` ($T_0=30,\ T_{\text{mult}}=2$): half-cosine
decay then restart, each cycle twice as long. Unit-norm gradient clipping bounds the
step in the curved AVM geometry. Early stopping on $\mathcal{V}$ after a warmup. When
`ema_centroids`, prototype positions are nudged toward their (soft-assigned) class
means each epoch; because that move is discontinuous, the optimiser's momentum state
for the centroid parameter is reset afterward so stale Adam moments don't fight it.

---

## 15. The cross-feature picture

Per-feature importance is a **diagonal** metric — it stretches axes only.
Cross-feature importance needs **off-diagonal** structure: the precision matrix
$\Sigma^{-1}$, whose eigenvectors are learned feature *combinations* and whose
inverse eigenvalues are their importances. The prototype mixture extends one
ellipsoid to a piecewise-quadric; the Fisher head supplies global discriminative
combinations. All of this captures interactions *to degree two*.

For explicitly nonlinear, higher-degree interactions there are two implemented /
designed steps:

- **Implemented probe** — the `'interaction'` KNN branch (§4): a fixed
  random-projection of all pairwise products, $z$-scored, dropped into the frozen
  vote. It answers *whether* interactions carry exploitable signal (watch whether the
  KNN head stops being down-weighted), at near-zero cost, without touching the AVM
  covariance.
- **Sketched, not integrated** — *free-coordinate interaction prototypes*. A
  prototype today is a point in raw space, so any product coordinate it carries is
  the product of its marginals — it cannot represent "the product matters but the
  marginals are average." Giving each prototype a *decoupled* learnable position in
  interaction-projection space lifts that constraint, at the cost of the
  "prototype = readable exemplar" property and extra covariance dimensions. Paired
  with a *learned sparse projection* (a bandit/successive-halving search over the
  $\binom{F}{2}$ pairs), this becomes a factorization-machine-style interaction head
  living inside the prototype geometry — interpretable about *which* pairs matter.

The natural unifying step is a single learned low-rank projection $L$ scored in
$Lx$-space, generalising the diagonal feature weights of §2 to feature *combinations*
directly — learned from the objective, stable under imbalance, with the rows of $L$
readable as the engineered features the model chose.
