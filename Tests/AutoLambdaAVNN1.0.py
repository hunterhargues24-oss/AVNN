#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
# from auto_lambda_avnn import AutoLambdaAVNN

def cv_run(ds_name, X, y, feature_names, model_kwargs):
    X = X.astype(np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, macros, weighteds = [], [], []
    last_model = None

    for train_idx, test_idx in skf.split(X, y):
        model = AutoLambdaAVNN(**model_kwargs)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        macros.append(f1_score(y[test_idx], pred, average='macro', zero_division=0))
        weighteds.append(f1_score(y[test_idx], pred, average='weighted', zero_division=0))
        last_model = model

    print(f"\n{'='*60}")
    print(f"{ds_name}  ({X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes)")
    print(f"{'='*60}")
    print(f"  Accuracy    : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Macro F1    : {np.mean(macros):.4f} ± {np.std(macros):.4f}")
    print(f"  Weighted F1 : {np.mean(weighteds):.4f} ± {np.std(weighteds):.4f}")
    if last_model is not None:
        print("\nModel summary (last fold):")
        print(last_model.summary(feature_names))

def holdout_run(ds_name, X, y, model_kwargs, subsample=None):
    X = X.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    if subsample and len(X_tr) > subsample:
        print(f"  Subsampling training to {subsample} (stratified)...")
        rng = np.random.RandomState(42)
        parts_X, parts_y = [], []
        for c in np.unique(y_tr):
            idx = np.where(y_tr == c)[0]
            n = max(1, int(subsample * len(idx) / len(y_tr)))
            sel = rng.choice(idx, n, replace=False)
            parts_X.append(X_tr[sel])
            parts_y.append(y_tr[sel])
        X_tr = np.vstack(parts_X)
        y_tr = np.concatenate(parts_y)

    print(f"\n{'='*60}")
    print(f"{ds_name}  (train {X_tr.shape}, test {X_te.shape}, {len(np.unique(y))} classes)")
    print(f"{'='*60}")

    model = AutoLambdaAVNN(**model_kwargs)
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
    print("\nModel summary:")
    print(model.summary())

if __name__ == "__main__":
    base_kwargs = {
        'k': 5,
        'lr': 1e-3,
        'epochs': 300,
        'batch_size': 64,
        'patience': 60,
        'val_fraction': 0,
        'label_smoothing': 0.0,
        'weight_cap': 1.5,
        'weight_decay': 1e-4,
        'feat_weight_reg': 0.05,
        'gravity': 0.1,
        'gravity_cap': 1.0,
        'lam_init': 0.7,
        'auto_lambda': False,
        'use_ivf': True,
        'verbose': False,
        'random_state': 42
    }

    # ---- Iris ----
    data = load_iris()
    cv_run("Iris", data.data, data.target, data.feature_names, base_kwargs)

    # ---- Wine ----
    data = load_wine()
    cv_run("Wine", data.data, data.target, data.feature_names, base_kwargs)

    # ---- Breast Cancer ----
    data = load_breast_cancer()
    cv_run("Breast Cancer", data.data, data.target, data.feature_names, base_kwargs)

    # ---- Red Wine Quality ----
    if os.path.exists("winequality-red.csv"):
        df = pd.read_csv("winequality-red.csv", sep=";")
        y = df["quality"].apply(lambda q: 0 if q <= 4 else (1 if q <= 6 else 2)).values
        X = df.drop("quality", axis=1).values
        cv_run("Red Wine Quality", X, y, list(df.columns[:-1]), base_kwargs)
    else:
        print("\nRed Wine Quality: winequality-red.csv not found – skipping.")

    # ---- White Wine Quality ----
    if os.path.exists("winequality-white.csv"):
        df = pd.read_csv("winequality-white.csv", sep=";")
        y = df["quality"].apply(lambda q: 0 if q <= 4 else (1 if q <= 6 else 2)).values
        X = df.drop("quality", axis=1).values
        cv_run("White Wine Quality", X, y, list(df.columns[:-1]), base_kwargs)
    else:
        print("\nWhite Wine Quality: winequality-white.csv not found – skipping.")

    # ---- ArrivalType ----
    if os.path.exists("ArrivalType.csv"):
        df = pd.read_csv("ArrivalType.csv")
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        X = df.drop(columns=["ArrivalType"]).values
        y = df["ArrivalType"].values
        large_kwargs = base_kwargs.copy()
        large_kwargs.update({'use_ivf': True, 'k': 10, 'gravity': 0.3})
        holdout_run("ArrivalType (oil/gas)", X, y, large_kwargs, subsample=200_000)
    else:
        print("\nArrivalType.csv not found – skipping large-dataset test.")
