"""
benchmark_small.py
5‑fold stratified cross‑validation on small sklearn datasets using static AVNNClassifier.
Saves results to CSV and generates a bar plot.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from avnn import AVNNClassifier   # adjust import to your package name

datasets = {
    "Iris": load_iris(),
    "Wine (cultivar)": load_wine(),
    "Breast Cancer": load_breast_cancer()
}

results = []

for name, data in datasets.items():
    X = data.data.astype(np.float32)
    y = data.target
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
df.to_csv("benchmark_small_results.csv", index=False)
print(df)

# Plot
plt.figure(figsize=(10,6))
x = np.arange(len(datasets))
width = 0.25
acc_vals = [float(r["Accuracy"].split()[0]) for r in results]
macro_vals = [float(r["Macro F1"].split()[0]) for r in results]
weighted_vals = [float(r["Weighted F1"].split()[0]) for r in results]

plt.bar(x - width, acc_vals, width, label='Accuracy')
plt.bar(x, macro_vals, width, label='Macro F1')
plt.bar(x + width, weighted_vals, width, label='Weighted F1')
plt.xticks(x, list(datasets.keys()), rotation=15)
plt.ylabel('Score')
plt.title('5‑fold CV Performance – Static AVNNClassifier')
plt.legend()
plt.tight_layout()
plt.savefig("benchmark_small_plot.png")
plt.show()
