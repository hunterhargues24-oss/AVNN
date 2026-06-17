# What the Model Is Trying to Do

Every classifier asks the same question: given a new sample, which class does it
most likely belong to? Most models draw lines or curves between classes. This model
instead asks: *how similar is this sample to each class, from several geometric
perspectives at once?* — and it learns not just where each class sits, but the shape
and orientation of each class's cluster.

## Step 1 — Normalisation

Every feature is scaled to fit between −1 and +1, so alcohol content and pH are
measured in the same geometric space and no feature dominates by its raw units.

## Step 2 — Two spaces, not one

The sample is transformed through **two separate sets of lenses** — a *learned* AVM
space and a *frozen* KNN space. These are kept apart on purpose: the AVM space
carries a learned notion of cluster shape (a covariance metric), while the KNN space
is a flat local view where extra perspectives can be added cheaply.

**AVM branches** (used to measure distance to class prototypes):

- **tanh_arccos** — how far a feature is from the middle of its range. Most
  sensitive in the centre, where subtle class differences often live. Usually the
  dominant branch.
- **boundary** — the only branch that looks at all features at once: where the whole
  point sits relative to the cube's corners. This is where cross-feature
  relationships enter.
- **circular** — how extreme a feature is regardless of direction: +0.9 and −0.9
  look the same to it.

**KNN branches** (used to find nearest neighbours):

- **linear** — raw signed position.
- **shape** — ignores absolute values; only the *relative profile* within a sample
  (which features are high vs low). Two wines with different absolute values but the
  same profile look identical here.
- **quadratic** — distance from zero with nonlinear emphasis at the extremes.
- *(optional)* **log / rank / clr** — heavy-tail, monotone-rank, and compositional
  views for harder data.
- *(optional)* **interaction** — looks at products of feature pairs, a probe for
  whether feature *combinations* (not single features) carry signal.

The branch weights are learned: the model discovers which perspectives matter for a
given dataset. On the oil-and-gas ArrivalType data, tanh_arccos earns nearly all the
weight; on wine cultivar, profile-shape matters more.

## Step 3 — AVM: how well does this fit each class?

Each class has learned **prototypes** — ideal points representing typical members,
one or several per class. The model measures distance from the sample to each
prototype, but not with a plain ruler: it uses a learned **covariance** for each
class, so a long, tilted cluster is measured along its own axes rather than as a
sphere. Closer to a prototype, under that class's shape → higher score.

By default the AVM score is a proper **probability** (a QDA posterior): it folds in
how common each class is and how spread-out its cluster is, not just raw closeness.
Two dials let you turn those extras down — useful on rare classes, where insisting
on the "how common" term would bury the minority. (A flatter "inverse-distance" mode
is also available when the downstream blending prefers softer, less spiky scores.)

AVM answers: *does this sample look like it belongs to class k, from a global,
shape-aware perspective?*

## Step 4 — KNN: who are the actual neighbours?

Completely separate from prototypes. In the KNN space the model finds the most
similar real training samples (via FAISS) and takes a distance-weighted vote.

KNN answers: *what classes do the actually-nearby training samples belong to?*

AVM and KNN often agree. Where they disagree is informative — AVM may be confused
near a boundary where two prototypes are equidistant, while KNN sees a clear local
majority; or KNN may be lost in a sparse region while AVM knows the point is
geometrically consistent with a class. Two optional heads can join the panel: a
**Fisher** head (the single most class-separating linear view) and a **geodesic**
head (which unfolds curved data before measuring).

## Step 5 — Fusion: how much to trust each answer

The heads' probabilities are blended. With just AVM and KNN this is a single weight
λ — chosen by grid search, or set per-sample by a **confidence gate**: whichever head
is more certain (a peakier probability distribution) leads. With more heads, the
model picks balanced or validation-tuned weights, defaulting to "treat them equally
unless the data clearly say otherwise." When two confident heads disagree, that
standoff is flagged for review.

## The full flow in one line each

1. Scale features to [−1, +1].
2. Transform through two spaces — learned AVM branches and frozen KNN branches.
3. AVM scores how shape-consistent the sample is with each class prototype (a QDA
   probability by default).
4. KNN counts what class the actual nearby samples are.
5. Fusion decides how much to trust each, globally or per-sample by confidence.

## Why it works well on imbalanced data

Rare classes fail different checks on different samples. Some minority samples sit
near a centroid but have the wrong feature profile; some have the right profile but
land in a sparse neighbourhood; some sit near the majority boundary where neighbours
mislead. No single view catches every failure. The shape-aware AVM metric, the local
KNN vote, the class-balanced training objective (you can bias early-stopping toward
macro-F1 and raise rare-class weight), and a blend that lets the most confident view
lead — together they hold where any one alone would slip.
