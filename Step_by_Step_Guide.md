# LearningAVNN — Step by Step

A plain-language walk through the pipeline, one step at a time.

1. **Normalise** each feature to a value between −1 and +1, so every feature lives
   in the same cube and none dominates by its raw scale.

2. **Two spaces, not one.** Transform each sample two separate ways:
   - a **learned AVM space** — three branches (angular position, cube-boundary
     coupling, symmetric extremeness) whose weights are trained;
   - a **frozen KNN space** — local views (signed position, within-sample profile,
     nonlinear extremeness, plus optional log / rank / compositional / pairwise-
     interaction branches).
   Keeping them separate lets the AVM space carry a cluster-shape metric while the
   KNN space stays a cheap flat chart for extra perspectives.

3. **Learn class prototypes** (ideal points, one or several per class) plus the
   branch weights, per-feature weights, per-class temperature, and — for each
   prototype — a **covariance** describing the shape and tilt of that class's
   cluster.

4. **Score the AVM head as a probability.** Measure distance from the sample to each
   prototype *under that class's covariance* (so a stretched, tilted cluster is
   measured along its own axes), then turn distances into class probabilities. By
   default this is a full **QDA posterior** that also accounts for how common each
   class is and how large its cluster is.
   - Two dials temper those extras: `prior_weight` (turn the "how common" term down
     to lift rare-class recall) and `logdet_weight` (turn the cluster-volume term
     down when data is scarce). A flatter `inverse_distance` scoring mode is also
     available.

5. **Compute KNN probabilities** separately — find the nearest real training samples
   in the KNN space (via FAISS) and take a distance-weighted vote.

6. **(Optional) add more heads** — a Fisher (LDA) head for the single most
   class-separating linear view, and/or a geodesic (Isomap) head that unfolds curved
   data before measuring.

7. **Fuse the heads** on the probability simplex — a single learned weight λ
   (AVM vs KNN), a per-sample confidence gate (the more certain head leads), or
   validation-tuned weights for larger head sets. Confident disagreements are
   flagged.

8. **Train** by minimising a composite geometric loss:
   - class-weighted classification error with label smoothing,
   - an ordinal Earth-Mover penalty for ordered classes,
   - a supervised-contrastive term pulling same-class points together,
   - branch-orthogonality and prototype-stability regularisers.
   Early stopping uses a metric you can bias from accuracy toward macro-F1
   (`val_macro_bias`).

9. **After training**, automatically choose the fusion scheme (and λ) that scores
   best on a held-out validation split.

10. **Predict** new data by applying the same two-space transforms, scoring both
    heads, and returning the fused probabilities.

11. **Inspect** what was learned: `summary()` shows branch/feature weights, per-class
    temperature, fusion choice, and prototype drift; `report(X, y)` gives per-class
    precision/recall/F1 and a confusion matrix and flags the lowest-F1 class — the
    targeted diagnostic for an imbalanced squeeze.
