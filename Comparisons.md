# LearningAVNN — Benchmark Comparison

Macro-F1 is the primary metric for imbalanced datasets; accuracy/weighted-F1 are
reported for context.

## Protocol & provenance (read first)

Three tiers of number appear below, kept separate on purpose:

- **LearningAVNN (current)** — the model in this repo, most recent run. 5-fold
  stratified CV on small/medium sets. These are the doc-7 results.
- **Tree baseline (measured here)** — `HistGradientBoostingClassifier`
  (`class_weight='balanced'`, early stopping), 5-fold stratified, `seed=42`.
  Run only on the datasets that ship with sklearn (Iris, Wine, Breast Cancer).
  These use a *different* fold seed than the LearningAVNN runs, so treat ±0.01 as
  noise; the controlled, identical-fold comparison is `bench_avnn_vs_tree.py`.
- **External (prior internal runs)** — XGBoost / RF / SVM / LogReg numbers for the
  wine-quality and ArrivalType sets, carried over from earlier internal CV. These
  have **not** yet been re-run on matched folds against the current model; the
  harness does that.

> **Binning caveat (White Wine).** The current runs bin white-wine quality as
> {3,4,5}/{6}/{7,8,9} → ~33/45/22. Earlier family benchmarks used a different
> binning (~4/75/22). White-wine macro-F1 across the two is therefore **not**
> directly comparable. Red Wine binning ({3,4}/{5,6}/{7,8} → ~4/82/14) is
> consistent.

---

## Reachable sets — LearningAVNN vs a real gradient-boosted tree

These three I could run a tree on directly. The point: a gradient-boosted tree is
the canonical interaction-modelling method, so if it can't beat LearningAVNN here,
there's no interaction headroom the geometric model is leaving on the table.

| Dataset | LearningAVNN macro | Tree (HGB) macro | Verdict |
|---------|--------------------|------------------|---------|
| Iris (150, 4f, balanced) | 0.9598 | 0.9666 | Tie — both at ceiling |
| Wine (178, 13f, balanced) | **0.9944** | 0.9565 | AVNN wins by ~3.8 pts |
| Breast Cancer (569, 30f, 37/63) | **0.9634** | 0.9493 | AVNN wins by ~1.4 pts |

On every reachable set, the tree ties (Iris) or loses (Wine, Breast Cancer). Wine
is a pure covariance/Gaussian-structure problem where the QDA work pays and trees
have nothing to add; on Breast Cancer the interaction-native method can't match the
per-prototype Mahalanobis. No reachable dataset shows interaction headroom AVNN is
missing.

---

## Imbalanced sets — current LearningAVNN

| Dataset | Accuracy | Macro F1 | Weighted F1 | Config note |
|---------|----------|----------|-------------|-------------|
| Red Wine (1599, 11f, ~4/82/14) | 0.8199 | **0.5953** | 0.8264 | `inverse_distance`, `prior_weight=0`, `weight_cap=2`, `val_macro_bias=0.85`, m=2 |
| White Wine (4898, 11f, ~33/45/22) | 0.6978 | **0.6950** | 0.6977 | `inverse_distance`, `prior_weight=0`, m=3 |

Macro-F1 here is gated by minority recall. The tuning that moved Red Wine — pushing
`val_macro_bias` to 0.85 and `weight_cap` to 2 (a *gentle* reweight; harder reweights
overcorrect on its ~44-sample minority) — is the lever, alongside `prior_weight=0`
to stop the class prior burying the rare class.

### External baselines on the imbalanced sets (prior internal runs — pending controlled rerun)

| Model | Red Wine macro | White Wine macro† |
|-------|----------------|-------------------|
| **LearningAVNN (current)** | **0.5953** | **0.6950** |
| XGBoost (default) | 0.527 | 0.638 |
| Random Forest (100) | 0.533 | 0.631 |
| Logistic Regression | 0.425 | 0.427 |
| SVM RBF | 0.301 | 0.285 |

†White Wine external numbers use the older ~4/75/22 binning — **not** comparable to
the current 0.6950 (different problem). Re-run with `bench_avnn_vs_tree.py` for a
matched comparison. On Red Wine (consistent binning), AVNN leads the field on macro;
SVM/LogReg collapse on imbalance without manual tuning.

### ArrivalType — Oil & Gas (proprietary, not re-run here)

Prior family result: ~0.58 macro-F1 at ~7s fit on a 200k subsample (training scales
with K, not N). The current LearningAVNN has not been run on ArrivalType in this
cycle (CSV absent from the test harness). This is the dataset where the QDA prior
may actually help — with ~87k minority rows the covariance and prior are
well-estimated, unlike the tiny wine minorities — so it warrants its own
`avm_score` / `prior_weight` sweep rather than inheriting the wine settings.

---

## Key findings

- **Trees reveal no interaction headroom on any reachable set.** AVNN ties on Iris
  and beats gradient-boosted trees on Wine and Breast Cancer. The interaction probe
  (`'interaction'` KNN branch) produced only small, consistent positive nudges
  (largest on the 30-feature Breast Cancer set, +0.0037 macro) — real, but not the
  tip of a missed-structure iceberg.
- **AVNN's edge is on imbalance and shape.** Where classes are correlated Gaussian
  blobs (Wine) or need shape-aware distance (Breast Cancer), the per-prototype
  Mahalanobis beats trees. Where minority recall is the bottleneck (Red Wine), the
  built-in class-balanced objective beats SVM/LogReg outright.
- **No per-dataset hyperparameter search required for the geometry** — defaults
  generalise; the imbalance levers (`weight_cap`, `val_macro_bias`, `prior_weight`)
  are the ones worth tuning, and they scale with how many minority samples exist.
- **The wine ceiling looks like noise, not missing structure** on the reachable
  evidence. The decisive test is the wine-quality tree row in
  `bench_avnn_vs_tree.py`: tree clearly above AVNN → real interaction signal, build
  the free-coordinate head; tree at/below AVNN → at the data's Bayes ceiling.

---

## AVNN family (historical context)

LearningAVNN is the PyTorch *learnable* lineage (listed as "BranchAdaptiveAVNN
(learnable)" in earlier benchmarks). Sibling implementations traded training for
speed:

| Variant | Training | Inference | FAISS | Best for |
|---------|----------|-----------|-------|----------|
| `AVNNClassifier` (static) | none | O(n·F) | no | fast baseline, small data |
| `LearningAVNN` (this repo) | PyTorch | O(K·m) score | optional | research, max accuracy, imbalance |
| `FastTriBranchAVNN` | none | O(k·log n) | yes | large balanced datasets |
| `TriAnchorAVNN` | none | O(k·log n) | yes | large imbalanced datasets |

Earlier family results referenced explicit "triangle"/"bilinear" pairwise scorers
and a post-hoc "gravity" prior. Both are superseded in the current model: the
per-prototype precision matrix's off-diagonals *are* the learned pairwise
interactions, and the class prior now lives inside the QDA score (`prior_weight`),
with `prior_temp` retained only as an optional post-fusion gravity (default 0).

---

## Computational characteristics

| Property | LearningAVNN | XGBoost / LightGBM | Deep nets (MLP/TabNet) |
|----------|--------------|--------------------|------------------------|
| Core learnable parameters | tens (branch/feature gates, τ) + K·m·F prototypes | N/A (trees) | 10k – 1M+ |
| Training scales with | K (centroids), not N | N | N |
| Inference scales with | K·m (AVM) + log N (FAISS KNN) | trees × depth | N × layers |
| Imbalance handling | built-in (balanced NLL, ordinal EMD, macro-biased selection, prior dials) | needs SMOTE / `scale_pos_weight` | needs class weights / oversampling |
| Interpretability | feature & branch weights, τ, per-class covariance, `report()` | SHAP post-hoc | SHAP, limited |
| Hyperparameter sensitivity | low — geometric defaults generalise | high | very high |

Approximate fit time: trees win on small data (XGBoost ~0.3s vs AVNN ~8–15s — the
~90 epochs × 5 mini-batches overhead is real at that scale). At 200k samples the
O(K) training flips the order (AVNN ~7s, the FAISS index build being the only step
touching all N).

### Honest limitations

- 30–50× slower than XGBoost on small datasets; epoch count dominates.
- CPU-only today (no GPU path).
- FAISS IVF needs a minimum sample count to train the index (flat fallback below it).
- No native categorical handling (requires encoding).
- The QDA volume term can be unstable at high D / low n — hence the `logdet_weight`
  dial; on tiny minorities the prior hurts macro-F1, hence `prior_weight`.

---

*Reproduce the controlled comparison with `bench_avnn_vs_tree.py` (identical folds,
both models, all five datasets). External baselines marked "prior internal runs"
have not yet been re-run on matched folds against the current model.*
