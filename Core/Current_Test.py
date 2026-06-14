"""
Benchmark LearningAVNN on six datasets.

Reflects the current architecture: split AVM (learned) / KNN (frozen) feature
spaces, configurable fusion heads (avm, knn, fisher, geodesic) blended by an
N-head gate, configurable KNN branches (linear/shape/quadratic + optional
log/rank/clr), per-class tau, EMA prototypes, RDA Mahalanobis. (Triangle
scoring was retired.)

Per-dataset configs below enable the additions where they helped in testing:
the Fisher head broadly, and the enriched KNN branches (log+rank) on the
imbalanced sets. Geodesic is available (heads=(...,'geodesic')) but left off —
it's the heaviest head and least validated.

Notebook use: assumes LearningAVNN is importable or already defined.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    from LearningAVNN import LearningAVNN
except Exception:
    pass  # assume LearningAVNN is already defined in the namespace

# Enriched KNN branch set: originals + monotone views (log, rank).
# Add 'clr' only for genuinely compositional features.
ENRICHED_KNN = ('linear', 'shape', 'quadratic', 'log', 'rank')

# ----------------------------------------------------------------------
# Helper: cross-validation for small/medium datasets
# ----------------------------------------------------------------------
def cv_run(ds_name, X, y, feature_names, model_kwargs):
    X = X.astype(np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, macs, wf1s = [], [], []
    last_model = None
    for train_idx, test_idx in skf.split(X, y):
        model = LearningAVNN(**model_kwargs)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        macs.append(f1_score(y[test_idx], pred, average='macro', zero_division=0))
        wf1s.append(f1_score(y[test_idx], pred, average='weighted', zero_division=0))
        last_model = model

    print(f"\n{'='*60}")
    print(f"{ds_name}  ({X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes)")
    print(f"{'='*60}")
    print(f"  Accuracy    : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Macro F1    : {np.mean(macs):.4f} ± {np.std(macs):.4f}")
    print(f"  Weighted F1 : {np.mean(wf1s):.4f} ± {np.std(wf1s):.4f}")
    if last_model:
        print("\nModel summary (last fold):")
        print(last_model.summary(feature_names))

# ----------------------------------------------------------------------
# Helper: hold-out for large dataset (ArrivalType)
# Bug fixes vs original:
#   1. subsample unpacking was inverted — now correctly keeps the
#      train_size slice, not the remainder
#   2. feature_names passed through to summary()
#   3. use_ivf defaulted to True for large datasets
# ----------------------------------------------------------------------
def holdout_run(ds_name, X, y, feature_names, model_kwargs, subsample=None):
    X = X.astype(np.float32)
    if subsample and len(X) > subsample:
        # Fix: keep the first split (size=subsample), discard the rest
        X, _, y, _ = train_test_split(
            X, y, train_size=subsample, stratify=y, random_state=42)
        print(f"  Subsampled to {len(X)} samples (stratified).")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"\n{'='*60}")
    print(f"{ds_name}  (train {X_tr.shape}, test {X_te.shape}, {len(np.unique(y))} classes)")
    print(f"{'='*60}")

    model = LearningAVNN(**model_kwargs)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_t = time.time() - t0
    t0 = time.time()
    pred = model.predict(X_te)
    pred_t = time.time() - t0

    print(f"  Fit time    : {fit_t:.2f}s")
    print(f"  Predict time: {pred_t:.2f}s")
    print(f"  Accuracy    : {accuracy_score(y_te, pred):.4f}")
    print(f"  Macro F1    : {f1_score(y_te, pred, average='macro', zero_division=0):.4f}")
    print(f"  Weighted F1 : {f1_score(y_te, pred, average='weighted', zero_division=0):.4f}")
    print("\nClassification report:")
    print(classification_report(y_te, pred, zero_division=0))
    print("\nModel summary:")
    print(model.summary(feature_names))  # Fix: pass feature_names

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # Base hyperparameters (conservative). New knobs:
    #   heads        : fusion heads — ('avm','knn'[,'fisher'][,'geodesic'])
    #   knn_branches : KNN-space branch set (Euclidean head; cheap to extend)
    # Base keeps the original AVM+KNN baseline; each dataset opts into the
    # additions explicitly below so the choices are visible.
    base_kwargs = {
        'k': 5,
        'lr': 1e-3,
        'epochs': 300,
        'batch_size': 256,
        'patience': 60,
        'val_fraction': 0.15,
        'label_smoothing': 0.05,
        'weight_cap': 1.5,
        'weight_decay': 1e-4,
        'feat_weight_reg': 0.05,
        'centroid_reg': 0.1,
        'centroid_sep': 0.05,
        'n_prototypes': 1,
        'intra_sep': 0.05,
        'ortho_reg': 0.00,
        'mahalanobis': True,
        'mahal_reg': 1e-6,
        'mahal_alpha': 0.5,
        'lam_floor': 0.5,
        'entropy_lambda': False,
        'lam_confident': 0.90,
        'lam_uncertain': 0.40,
        'ordinal': False,             # set explicitly per dataset
        'ordinal_weight': 0.5,        # Fix: explicit — was implicit model default
        'per_class_tau': True,
        'val_macro_bias': 0.5,
        'use_ivf': False,
        'supcon': False,
        'supcon_weight': 0.3,
        'supcon_temp': 0.1,
        'ema_centroids': True,
        'ema_beta': 0.9,
        'heads': ('avm', 'knn'),                                # NEW
        'knn_branches': ('linear', 'shape', 'quadratic'),       # NEW
        'random_state': 42,
        'verbose': False,
        'use_dual_boundary': False,
        'boundary_init': True
    }

    # ------------------------------------------------------------------
    # Iris  — clean/balanced: add Fisher head; keep original KNN branches
    # ------------------------------------------------------------------
    from sklearn.datasets import load_iris
    data = load_iris()
    kw = base_kwargs.copy()
    kw.update({'ordinal': False, 'supcon': True, 'ema_centroids': False,
               'heads': ('avm', 'knn', 'fisher')})
    cv_run("Iris", data.data, data.target, list(data.feature_names), kw)

    # ------------------------------------------------------------------
    # Wine (cultivar) — clean/balanced: add Fisher head
    # ------------------------------------------------------------------
    from sklearn.datasets import load_wine
    data = load_wine()
    kw = base_kwargs.copy()
    kw.update({'ordinal': False, 'supcon': True, 'ema_centroids': False,
               'heads': ('avm', 'knn', 'fisher')})
    cv_run("Wine", data.data, data.target, list(data.feature_names), kw)

    # ------------------------------------------------------------------
    # Breast Cancer — Fisher head + enriched KNN (fisher+enriched best in test)
    # ------------------------------------------------------------------
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    kw = base_kwargs.copy()
    kw.update({
        'ordinal': True,
        'ordinal_weight': 0.5,
        'supcon': True,
        'ema_centroids': False,
        'val_macro_bias': 0.7,
        'heads': ('avm', 'knn', 'fisher'),
        'knn_branches': ENRICHED_KNN,
    })
    cv_run("Breast Cancer", data.data, data.target, list(data.feature_names), kw)

    # ------------------------------------------------------------------
    # Red Wine Quality (ordered, imbalanced: ~4% / 82% / 14%)
    #   imbalanced+multimodal: Fisher head + enriched KNN (log/rank helped most)
    # ------------------------------------------------------------------
    if os.path.exists("winequality-red.csv"):
        df = pd.read_csv("winequality-red.csv", sep=";")
        y = df["quality"].apply(
            lambda q: 0 if q <= 4 else (1 if q <= 6 else 2)).values
        X = df.drop("quality", axis=1).values.astype(np.float32)
        feature_names = list(df.columns[:-1])
        kw = base_kwargs.copy()
        kw.update({
            'ordinal': True,
            'ordinal_weight': 0.5,
            'supcon': True,
            'ema_centroids': True,
            'n_prototypes': 2,
            'val_macro_bias': 0.8,
            'heads': ('avm', 'knn'),
            'knn_branches': ENRICHED_KNN,
            # NOTE: intra_sep is now a live gradient (was inert). Re-sweep it
            # for this multi-prototype config — 0.05 may no longer be optimal.
        })
        cv_run("Red Wine Quality", X, y, feature_names, kw)
    else:
        print("\nRed Wine Quality: winequality-red.csv not found – skipping.")

    # ------------------------------------------------------------------
    # White Wine Quality (ordered, larger: ~34% / 45% / 21%)
    # ------------------------------------------------------------------
    if os.path.exists("winequality-white.csv"):
        df = pd.read_csv("winequality-white.csv", sep=";")
        y = df["quality"].apply(
            lambda q: 0 if q <= 5 else (1 if q == 6 else 2)).values
        X = df.drop("quality", axis=1).values.astype(np.float32)
        feature_names = list(df.columns[:-1])
        kw = base_kwargs.copy()
        kw.update({
            'ordinal': True,
            'ordinal_weight': 0.5,
            'supcon': True,
            'ema_centroids': True,
            'n_prototypes': 1,
            'val_macro_bias': 0.7,
            'heads': ('avm', 'knn', 'fisher'),
            'knn_branches': ENRICHED_KNN,
        })
        cv_run("White Wine Quality", X, y, feature_names, kw)
    else:
        print("\nWhite Wine Quality: winequality-white.csv not found – skipping.")

    # ------------------------------------------------------------------
    # ArrivalType (large, severely imbalanced)
    #   the target regime: Fisher head + enriched KNN (log/rank gave the
    #   biggest single lift on the ArrivalType-like synthetic set)
    # ------------------------------------------------------------------
    if os.path.exists("ArrivalType.csv"):
        df = pd.read_csv("ArrivalType.csv")
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        X = df.drop(columns=["ArrivalType"]).values.astype(np.float32)
        y = df["ArrivalType"].values
        feature_names = list(df.columns[df.columns != "ArrivalType"])
        kw = base_kwargs.copy()
        kw.update({
            'k': 10,
            'epochs': 200,
            'batch_size': 512,
            'patience': 40,
            'val_fraction': 0.05,
            'mahal_alpha': 0.2,
            'ordinal': False,
            'supcon': False,
            'ema_centroids': True,
            'n_prototypes': 2,
            'val_macro_bias': 0.9,
            'use_ivf': True,          # Fix: True for large dataset
            'entropy_lambda': False,
            'lam_confident': 0.80,
            'lam_uncertain': 0.30,
            'heads': ('avm', 'knn', 'fisher'),
            'knn_branches': ENRICHED_KNN,
            'verbose': True,
        })
        holdout_run("ArrivalType (oil/gas)", X, y, feature_names, kw,
                    subsample=200_000)
    else:
        print("\nArrivalType.csv not found – skipping large-dataset test.")


if __name__ == "__main__":
    main()
