"""
Benchmark LearningAVNN on standard datasets.

Datasets:
  1. Iris            (sklearn bundled)
  2. Wine            (sklearn bundled)
  3. Breast Cancer   (sklearn bundled)
  4. Red Wine Quality   (UCI, downloaded if not local)
  5. White Wine Quality (UCI, downloaded if not local)

Config: matches what we landed on for ArrivalType, with per-dataset
overrides for the imbalanced wine quality datasets.

Usage:
  python benchmark_all.py
"""

import os
import time
import urllib.request
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)

# from learning_avnn import LearningAVNN


# ── Data loaders ─────────────────────────────────────────────────────────────

UCI_WINE = ("https://archive.ics.uci.edu/ml/"
            "machine-learning-databases/wine-quality")


def _bin_quality_red(q):
    """Bins matching the original red wine benchmark distribution
    (≈4% / 82% / 14%)."""
    if q <= 4:
        return 0
    if q <= 6:
        return 1
    return 2


def _bin_quality_white(q):
    """Bins matching the original white wine benchmark distribution
    (≈33% / 45% / 22%)."""
    if q <= 5:
        return 0
    if q == 6:
        return 1
    return 2


def _load_wine_quality(color, cache_dir="./data"):
    """Load red or white wine quality. Caches to disk after first download."""
    import pandas as pd
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, f"winequality-{color}.csv")
    if not os.path.exists(local):
        url = f"{UCI_WINE}/winequality-{color}.csv"
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, local)
    df = pd.read_csv(local, sep=';')
    binner = _bin_quality_red if color == 'red' else _bin_quality_white
    y = df['quality'].apply(binner).values.astype(np.int64)
    X = df.drop('quality', axis=1).values.astype(np.float32)
    names = df.columns[:-1].tolist()
    return X, y, names


# ── Benchmark runner ─────────────────────────────────────────────────────────

def benchmark(name, X, y, feature_names, config, n_splits=5,
              random_state=42, show_per_class=False):
    """Run k-fold CV and print results in the standard benchmark format."""
    K = len(np.unique(y))
    counts = np.bincount(y, minlength=K)
    pct = counts / counts.sum() * 100

    print("=" * 60)
    print(f"{name}  ({len(X):,} samples, {X.shape[1]} features, "
          f"{K} classes)")
    print("=" * 60)
    print(f"  class distribution:")
    for c, n, p in zip(range(K), counts, pct):
        print(f"    {c}: {n:,}  ({p:.1f}%)")

    accs, macros, weighteds = [], [], []
    all_y_true, all_y_pred = [], []
    last_model = None
    t0 = time.time()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        model = LearningAVNN(**config)
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[te_idx])
        accs.append(accuracy_score(y[te_idx], pred))
        macros.append(f1_score(y[te_idx], pred, average='macro'))
        weighteds.append(f1_score(y[te_idx], pred, average='weighted'))
        all_y_true.append(y[te_idx])
        all_y_pred.append(pred)
        last_model = model

    elapsed = time.time() - t0

    print()
    print(f"  Accuracy    : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Macro F1    : {np.mean(macros):.4f} ± {np.std(macros):.4f}")
    print(f"  Weighted F1 : {np.mean(weighteds):.4f} ± {np.std(weighteds):.4f}")
    print(f"  Total time  : {elapsed:.1f}s  ({elapsed/n_splits:.1f}s per fold)")

    if show_per_class:
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        print()
        print(f"  Per-class metrics (aggregated across folds):")
        print(classification_report(y_true, y_pred, digits=4))
        print(f"  Confusion matrix (rows=true, cols=pred):")
        cm = confusion_matrix(y_true, y_pred)
        header = "        " + "".join(f"{c:>8}" for c in range(K))
        print(header)
        for i, row in enumerate(cm):
            print(f"  {i:>4}  " + "".join(f"{v:>8}" for v in row))

    print()
    print("Model summary (last fold):")
    print(last_model.summary(feature_names=feature_names))
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Shared config — everything that should be consistent across datasets
    base_config = dict(
        branch_set='extended',
        device='auto',
        random_state=42,
        verbose=False
    )

    # Balanced datasets: minimal config, n_prototypes=1, no imbalance handling
    balanced_config = dict(base_config)
    balanced_config.update(
        n_prototypes=1,
        ema_centroids=False,
        weight_cap=1.5,
    )

    # Imbalanced wine quality: more prototypes, EMA on, macro-biased early
    # stopping, higher weight cap
    imbalanced_config = dict(base_config)
    imbalanced_config.update(
        n_prototypes=1,
        ema_centroids=True,
        entropy_lambda=False,
        weight_cap=6.0,
        val_macro_bias=0.85,
        supcon=False,
        ordinal=False, 
        mahalanobis=True # Wine quality IS ordinal, but ordinal=True hasn't
                          # helped in practice on these — keep it off for now
    )

    # sklearn-bundled datasets
    iris = load_iris()
    wine = load_wine()
    bc   = load_breast_cancer()

    benchmark("Iris", iris.data, iris.target,
              feature_names=iris.feature_names,
              config=balanced_config)

    benchmark("Wine", wine.data, wine.target,
              feature_names=wine.feature_names,
              config=balanced_config)

    benchmark("Breast Cancer", bc.data, bc.target,
              feature_names=bc.feature_names,
              config=imbalanced_config,
              show_per_class=True)

    # Wine quality (downloaded / cached)
    try:
        rX, ry, rnames = _load_wine_quality('red')
        benchmark("Red Wine Quality", rX, ry,
                  feature_names=rnames,
                  config=imbalanced_config,
                  show_per_class=True)
    except Exception as e:
        print(f"Red Wine Quality skipped: {e}\n")

    try:
        wX, wy, wnames = _load_wine_quality('white')
        benchmark("White Wine Quality", wX, wy,
                  feature_names=wnames,
                  config=imbalanced_config,
                  show_per_class=True)
    except Exception as e:
        print(f"White Wine Quality skipped: {e}\n")


if __name__ == "__main__":
    main()
