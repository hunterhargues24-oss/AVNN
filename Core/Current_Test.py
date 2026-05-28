"""
Benchmark LearningAVNN (split AVM/KNN spaces, optional triangle scoring,
entropy lambda, per‑class tau, etc.) on six datasets.
"""

import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
# from learningavnn import LearningAVNN   # your model class

# ----------------------------------------------------------------------
# Helper: cross‑validation for small/medium datasets
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
# Helper: hold‑out for large dataset (ArrivalType)
# ----------------------------------------------------------------------
def holdout_run(ds_name, X, y, model_kwargs, subsample=None):
    X = X.astype(np.float32)
    if subsample and len(X) > subsample:
        _, X, _, y = train_test_split(
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
    print(model.summary())

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Base hyperparameters (conservative, works for most datasets)
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
        'use_triangle': False,
        'tri_weight': 0.3,
        'ortho_reg': 0.00,
        'mahalanobis': True,          # full RDA
        'mahal_reg': 1e-6,
        'mahal_alpha': 0.5,
        'lam_floor': 0.3,
        'entropy_lambda': False,
        'lam_confident': 0.90,
        'lam_uncertain': 0.40,
        'ordinal': True,              # set per dataset (only if ordered)
        'per_class_tau': True,
        'val_macro_bias': 0.5,        # 0 = pure (acc+weighted)/2, 1 = pure macro
        'use_ivf': False,             # set True for large data
        'supcon': False,
        'supcon_weight': 0.3,
        'supcon_temp': 0.1,
        'ema_centroids': True,
        'ema_beta': 0.9,
        'random_state': 42,
        'verbose': False,
        'use_dual_boundary': False,
    }

    # ------------------------------------------------------------------
    # Iris
    # ------------------------------------------------------------------
    from sklearn.datasets import load_iris
    data = load_iris()
    kw = base_kwargs.copy()
    kw.update({'ordinal': False, 'supcon': True, 'ema_centroids': False})
    cv_run("Iris", data.data, data.target, data.feature_names, kw)

    # ------------------------------------------------------------------
    # Wine (cultivar)
    # ------------------------------------------------------------------
    from sklearn.datasets import load_wine
    data = load_wine()
    kw = base_kwargs.copy()
    kw.update({'ordinal': False, 'supcon': True, 'ema_centroids': False})
    cv_run("Wine", data.data, data.target, data.feature_names, kw)

    # ------------------------------------------------------------------
    # Breast Cancer
    # ------------------------------------------------------------------
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    kw = base_kwargs.copy()
    kw.update({'ordinal': True, 'supcon': True, 'ema_centroids': False,
               'val_macro_bias': 0.5})      # favour macro F1
    cv_run("Breast Cancer", data.data, data.target, data.feature_names, kw)

    # ------------------------------------------------------------------
    # Red Wine Quality (ordered, imbalanced)
    # ------------------------------------------------------------------
    if os.path.exists("winequality-red.csv"):
        df = pd.read_csv("winequality-red.csv", sep=";")
        y = df["quality"].apply(lambda q: 0 if q <= 4 else (1 if q <= 6 else 2)).values
        X = df.drop("quality", axis=1).values.astype(np.float32)
        feature_names = list(df.columns[:-1])
        kw = base_kwargs.copy()
        kw.update({
            'ordinal': True,
            'supcon': True,
            'ema_centroids': True,
            'use_triangle': False,
            'tri_weight': 0.3,
            'n_prototypes': 2,
            'val_macro_bias': 0.5,
        })
        cv_run("Red Wine Quality", X, y, feature_names, kw)
    else:
        print("\nRed Wine Quality: winequality-red.csv not found – skipping.")

    # ------------------------------------------------------------------
    # White Wine Quality (ordered, larger)
    # ------------------------------------------------------------------
    if os.path.exists("winequality-white.csv"):
        df = pd.read_csv("winequality-white.csv", sep=";")
        y = df["quality"].apply(lambda q: 0 if q <= 5 else (1 if q == 6 else 2)).values
        X = df.drop("quality", axis=1).values.astype(np.float32)
        feature_names = list(df.columns[:-1])
        kw = base_kwargs.copy()
        kw.update({
            'ordinal': True,
            'supcon': True,
            'ema_centroids': True,
            'use_triangle': False,
            'tri_weight': 0.3,
            'n_prototypes': 1,
            'val_macro_bias': 0.5,
        })
        cv_run("White Wine Quality", X, y, feature_names, kw)
    else:
        print("\nWhite Wine Quality: winequality-white.csv not found – skipping.")

    # ------------------------------------------------------------------
    # ArrivalType (large, severely imbalanced)
    # ------------------------------------------------------------------
    if os.path.exists("ArrivalType.csv"):
        df = pd.read_csv("ArrivalType.csv")
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        X = df.drop(columns=["ArrivalType"]).values.astype(np.float32)
        y = df["ArrivalType"].values
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        kw = base_kwargs.copy()
        kw.update({
            'k': 10,
            'epochs': 200,
            'batch_size': 512,
            'patience': 40,
            'val_fraction': 0.05,
            'mahal_alpha': 0.2,          # less regularisation for severe imbalance
            'ordinal': False,
            'supcon': False,
            'ema_centroids': True,
            'use_triangle': False,
            'n_prototypes': 2,
            'val_macro_bias': 0.9,        # strongly favour macro F1
            'use_ivf': False,
            'entropy_lambda': False,       # use confidence‑weighted lambda
            'lam_confident': 0.80,
            'lam_uncertain': 0.30,
            'verbose': True,
        })
        holdout_run("ArrivalType (oil/gas)", X, y, kw, subsample=200_000)
    else:
        print("\nArrivalType.csv not found – skipping large-dataset test.")
