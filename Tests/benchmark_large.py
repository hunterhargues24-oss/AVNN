"""
benchmark_large.py
Holdout evaluation on the large ArrivalType dataset using FastTriBranchAVNN.
Subsamples training to 200,000 points. Saves results to CSV and plots.
"""

import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from fast_tri_branch_avnn import FastTriBranchAVNN   # adjust import

DATA_PATH = "ArrivalType.csv"
TARGET_COL = "ArrivalType"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SUBSAMPLE_TRAIN = 200000   # set to None to use full training

# Load data
df = pd.read_csv(DATA_PATH)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
y = df[TARGET_COL].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# Subsample training if needed
if SUBSAMPLE_TRAIN and len(X_train) > SUBSAMPLE_TRAIN:
    print(f"Subsampling training to {SUBSAMPLE_TRAIN} (stratified)...")
    parts_X, parts_y = [], []
    for c in np.unique(y_train):
        idx = np.where(y_train == c)[0]
        n = max(1, int(SUBSAMPLE_TRAIN * len(idx) / len(y_train)))
        sel = np.random.RandomState(RANDOM_STATE).choice(idx, n, replace=False)
        parts_X.append(X_train[sel]); parts_y.append(y_train[sel])
    X_train = np.vstack(parts_X); y_train = np.concatenate(parts_y)

# Model
model = FastTriBranchAVNN(lam=0.7, k=10, w_ang=0.34, w_euc=0.33, w_shape=0.33,
                          transform='tanh_ac', norm_range='[-1,1]',
                          use_ivf=True, nlist=1000, nprobe=10)

print("Fitting model...")
t0 = time.time()
model.fit(X_train, y_train)
fit_time = time.time() - t0

print("Predicting...")
t0 = time.time()
y_pred = model.predict(X_test)
pred_time = time.time() - t0

acc = accuracy_score(y_test, y_pred)
macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Save results
results = {
    "Dataset": "ArrivalType",
    "Accuracy": f"{acc:.4f}",
    "Macro F1": f"{macro:.4f}",
    "Weighted F1": f"{weighted:.4f}",
    "Fit time (s)": f"{fit_time:.2f}",
    "Predict time (s)": f"{pred_time:.2f}",
    "Training samples": len(X_train),
    "Test samples": len(X_test)
}
df_results = pd.DataFrame([results])
df_results.to_csv("benchmark_large_results.csv", index=False)
print(df_results.to_string())

# Simple bar plot of metrics
plt.figure(figsize=(6,4))
metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
values = [acc, macro, weighted]
plt.bar(metrics, values, color=['blue', 'orange', 'green'])
plt.ylim(0, 1)
plt.title('FastTriBranchAVNN on ArrivalType')
plt.ylabel('Score')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
plt.savefig("benchmark_large_plot.png")
plt.show()
