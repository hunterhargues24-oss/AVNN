"""
benchmark_wine.py
5‑fold CV on red and white wine datasets using static AVNNClassifier.
Saves results to CSV and generates a bar plot.
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from avnn import AVNNClassifier   # adjust import

def load_wine_data(path, bin_quality=True):
    df = pd.read_csv(path, sep=';')
    if bin_quality:
        y = df["quality"].apply(lambda q: 0 if q <= 4 else (1 if q <= 6 else 2)).values
    else:
        y = df["quality"].values
    X = df.drop("quality", axis=1).values.astype(np.float32)
    # For red wine, use pruned features (VA, alcohol, sulphates, pH)
    if "red" in path:
        X = X[:, [1, 10, 9, 8]]
    return X, y

wine_files = {
    "Red Wine Quality": "winequality-red.csv",
    "White Wine Quality": "winequality-white.csv"
}

results = []

for name, fname in wine_files.items():
    if not os.path.exists(fname):
        print(f"{fname} not found, skipping {name}")
        continue
    X, y = load_wine_data(fname)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, macros, weighteds = [], [], []
    for tr, te in skf.split(X, y):
        model = AVNNClassifier(alpha=0.5, lam=0.7, k=5,
                               transform='tanh_ac', norm_range='[-1,1]')
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        macros.append(f1_score(y[te], pred, average='macro', zero_division=0))
        weighteds.append(f1_score(y[te], pred, average='weighted', zero_division=0))
    results.append({
        "Dataset": name,
        "Accuracy": f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
        "Macro F1": f"{np.mean(macros):.4f} ± {np.std(macros):.4f}",
        "Weighted F1": f"{np.mean(weighteds):.4f} ± {np.std(weighteds):.4f}"
    })

df = pd.DataFrame(results)
df.to_csv("benchmark_wine_results.csv", index=False)
print(df)

# Plot
if len(results) > 0:
    plt.figure(figsize=(8,6))
    x = np.arange(len(results))
    width = 0.25
    acc_vals = [float(r["Accuracy"].split()[0]) for r in results]
    macro_vals = [float(r["Macro F1"].split()[0]) for r in results]
    weighted_vals = [float(r["Weighted F1"].split()[0]) for r in results]
    plt.bar(x - width, acc_vals, width, label='Accuracy')
    plt.bar(x, macro_vals, width, label='Macro F1')
    plt.bar(x + width, weighted_vals, width, label='Weighted F1')
    plt.xticks(x, [r["Dataset"] for r in results], rotation=15)
    plt.ylabel('Score')
    plt.title('5‑fold CV Performance – Static AVNNClassifier (wine)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_wine_plot.png")
    plt.show()
