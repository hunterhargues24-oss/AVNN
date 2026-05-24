"""
FastTriBranchAVNN benchmark — six datasets.
Five sklearn-bundled + one large oil/gas dataset (ArrivalType.csv).
Run with: python test_fast_tri_branch.py
"""

import sys, time, os
import numpy as np
for key in [k for k in sys.modules if 'FastTriBranch' in k]:
    del sys.modules[key]

# from FastTriBranchAVNN import FastTriBranchAVNN   # your static three‑branch class

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd


def cv_benchmark(ds_name, X, y, feature_names, kwargs):
    """5-fold CV for small/medium datasets."""
    X = X.astype(np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, macros, weighteds = [], [], []
    last = None
    for tr, te in skf.split(X, y):
        # Static model – no training, just compute centroids and FAISS index
        last = FastTriBranchAVNN(**kwargs)
        last.fit(X[tr], y[tr])
        pred = last.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        macros.append(f1_score(y[te], pred, average='macro', zero_division=0))
        weighteds.append(f1_score(y[te], pred, average='weighted', zero_division=0))

    print(f"\n{'='*62}")
    print(f"{ds_name}  ({X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(set(y))} classes)")
    print(f"{'='*62}")
    print(f"  Accuracy    : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Macro F1    : {np.mean(macros):.4f} ± {np.std(macros):.4f}")
    print(f"  Weighted F1 : {np.mean(weighteds):.4f} ± {np.std(weighteds):.4f}")
    if last is not None:
        print()
        print(last.summary(feature_names))


def holdout_benchmark(ds_name, X, y, kwargs, subsample=None):
    """Single train/test split with timing — for the large dataset."""
    X = X.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    if subsample and len(X_tr) > subsample:
        print(f"  Subsampling training to {subsample} (stratified)...")
        parts_X, parts_y = [], []
        for c in np.unique(y_tr):
            idx = np.where(y_tr == c)[0]
            n   = max(1, int(subsample * len(idx) / len(y_tr)))
            sel = np.random.RandomState(42).choice(idx, n, replace=False)
            parts_X.append(X_tr[sel]); parts_y.append(y_tr[sel])
        X_tr = np.vstack(parts_X); y_tr = np.concatenate(parts_y)

    print(f"\n{'='*62}")
    print(f"{ds_name}  (train {X_tr.shape}, test {X_te.shape}, "
          f"{len(set(y))} classes)")
    print(f"{'='*62}")

    model = FastTriBranchAVNN(**kwargs)
    t0 = time.time(); model.fit(X_tr, y_tr)
    fit_t = time.time() - t0
    t0 = time.time(); pred = model.predict(X_te)
    pred_t = time.time() - t0

    print(f"  Fit time    : {fit_t:.2f}s")
    print(f"  Predict time: {pred_t:.2f}s")
    print(f"  Accuracy    : {accuracy_score(y_te, pred):.4f}")
    print(f"  Macro F1    : {f1_score(y_te, pred, average='macro', zero_division=0):.4f}")
    print(f"  Weighted F1 : {f1_score(y_te, pred, average='weighted', zero_division=0):.4f}")
    print()
    print(model.summary())


# ── small / medium datasets — 5‑fold CV ───────────────────────────────────────
# Common parameters for the static model (no training epochs/patience)
common = dict(
    lam=0.7, k=5,
    w_ang=0.34, w_euc=0.33, w_shape=0.33,
    feat_weights=None, transform='tanh_ac', norm_range='[-1,1]',
    use_ivf=True, nlist=1000, nprobe=10
)

ds = load_iris()
cv_benchmark("Iris", ds.data, ds.target, list(ds.feature_names), common)

ds = load_wine()
cv_benchmark("Wine (cultivar)", ds.data, ds.target, list(ds.feature_names), common)

ds = load_breast_cancer()
cv_benchmark("Breast Cancer", ds.data, ds.target, list(ds.feature_names), common)

# Red & white wine quality
for label, fname in [("Red Wine Quality", "winequality-red.csv"),
                     ("White Wine Quality", "winequality-white.csv")]:
    if os.path.exists(fname):
        df = pd.read_csv(fname, sep=";")
        y  = df["quality"].apply(
            lambda q: 0 if q <= 4 else (1 if q <= 6 else 2)).values
        X  = df.drop("quality", axis=1).values
        cv_benchmark(label, X, y, list(df.drop("quality", axis=1).columns),
                     dict(common, feat_weights=None))   # no weight_cap needed
    else:
        print(f"\n{label}: {fname} not found, skipping.")

# ── large oil/gas dataset — holdout with timing ───────────────────────────────
if os.path.exists("ArrivalType.csv"):
    df = pd.read_csv("ArrivalType.csv")
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    X = df.drop(columns=["ArrivalType"]).values
    y = df["ArrivalType"].values
    holdout_benchmark(
        "ArrivalType (oil/gas)", X, y,
        dict(lam=0.7, k=10,
             w_ang=0.34, w_euc=0.33, w_shape=0.33,
             feat_weights=None, transform='tanh_ac', norm_range='[-1,1]',
             use_ivf=True, nlist=1000, nprobe=10),
        subsample=200000)
else:
    print("\nArrivalType.csv not found, skipping large-dataset test.")
