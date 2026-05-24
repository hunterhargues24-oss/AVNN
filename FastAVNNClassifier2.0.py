"""
FastAVNNClassifier – FAISS‑accelerated KNN with 3 branches.
Optimised for speed on large datasets.
"""

import numpy as np
import faiss
from sklearn.base import BaseEstimator, ClassifierMixin

class FastAVNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.5, lam=0.7, k=5, transform='tanh_ac',
                 norm_range='[-1,1]', eps=1e-10, use_ivf=True, nlist=1000, nprobe=10,
                 w_ang=0.34, w_euc=0.33, w_shape=0.33):
        self.alpha = alpha          # kept for compatibility, not directly used
        self.lam = lam
        self.k = k
        self.transform = transform
        self.norm_range = norm_range
        self.eps = eps
        self.use_ivf = use_ivf
        self.nlist = nlist
        self.nprobe = nprobe
        # Normalise branch weights to sum to 1
        total = w_ang + w_euc + w_shape
        self.w_ang = w_ang / total
        self.w_euc = w_euc / total
        self.w_shape = w_shape / total

    # ---------- preprocessing ----------
    def _norm_fit(self, X):
        self.mn_ = X.min(axis=0)
        self.rng_ = np.maximum(X.max(axis=0) - self.mn_, 1e-10)

    def _norm_apply(self, X):
        x = (X - self.mn_) / self.rng_
        if self.norm_range == '[-1,1]':
            x = -1.0 + 2.0 * x
        return x.astype(np.float32)

    def _tanh_ac(self, x):
        if self.norm_range == '[-1,1]':
            x = np.clip(x, -1.0 + 1e-7, 1.0 - 1e-7)
            return np.tanh(np.arccos(x) * 0.8).astype(np.float32)
        else:
            x = np.clip(x, 0.0, 1.0)
            return np.tanh(np.arccos(x) * 1.6).astype(np.float32)

    def _get_transform_fn(self):
        if callable(self.transform):
            return self.transform
        transforms = {
            'tanh_ac': self._tanh_ac,
            'identity': lambda x: x,
            'tan': lambda x: np.tan(np.clip(x, 0, 1) * np.pi / 2 * 0.97),
            'neglog': lambda x: -np.log(np.clip(x, 1e-10, 1.0)),
        }
        return transforms[self.transform]

    def _prepare(self, X):
        X_norm = self._norm_apply(X)
        X_trans = self._get_transform_fn()(X_norm)
        return X_norm, X_trans

    def _angular(self, X_norm):
        return np.arccos(np.clip(X_norm, -1.0 + self.eps, 1.0 - self.eps))

    def _shape(self, X_norm):
        mu = X_norm.mean(axis=1, keepdims=True)
        sd = X_norm.std(axis=1, keepdims=True).clip(min=self.eps)
        return (X_norm - mu) / sd

    def _make_combined_vector(self, X_norm, X_trans):
        ang = self._angular(X_norm)
        euc = X_trans
        shape = self._shape(X_norm)
        scale_ang = np.sqrt(self.w_ang)
        scale_euc = np.sqrt(self.w_euc)
        scale_shape = np.sqrt(self.w_shape)
        return np.concatenate([scale_ang * ang, scale_euc * euc, scale_shape * shape], axis=1)

    # ---------- fit ----------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]

        self._norm_fit(X)
        X_norm, X_trans = self._prepare(X)
        combined = self._make_combined_vector(X_norm, X_trans)

        # Centroids in combined space
        self.centroids_ = np.array([combined[y == c].mean(axis=0) for c in self.classes_])

        # Build FAISS index (IVF by default for speed)
        if self.use_ivf:
            # Adjust nlist to be at most the number of training points
            nlist = min(self.nlist, combined.shape[0] // 39)
            if nlist < 1:
                nlist = 1
            quantizer = faiss.IndexFlatL2(combined.shape[1])
            index = faiss.IndexIVFFlat(quantizer, combined.shape[1], nlist, faiss.METRIC_L2)
            index.train(combined)
            index.nprobe = self.nprobe
        else:
            index = faiss.IndexFlatL2(combined.shape[1])
        index.add(combined)
        self.index_ = index
        self.y_train_ = y

        # Precompute class indices for fast voting
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        self.train_class_idx_ = np.array([self.class_to_idx_[c] for c in y])

        return self

    # ---------- predict ----------
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_norm, X_trans = self._prepare(X)
        combined = self._make_combined_vector(X_norm, X_trans)

        # AVM branch (centroids)
        d_cent = np.linalg.norm(combined[:, None, :] - self.centroids_[None, :, :], axis=2)
        raw_cent = 1.0 / (d_cent + self.eps)
        avm_proba = raw_cent / raw_cent.sum(axis=1, keepdims=True)

        # KNN branch – FAISS search
        distances, indices = self.index_.search(combined, self.k)
        weights = 1.0 / (distances + self.eps)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Fully vectorised voting (no Python loop per test point)
        n_test = combined.shape[0]
        n_classes = len(self.classes_)
        knn_proba = np.zeros((n_test, n_classes), dtype=np.float32)
        flat_indices = indices.ravel()
        flat_weights = weights.ravel()
        test_indices = np.repeat(np.arange(n_test), self.k)
        np.add.at(knn_proba, (test_indices, self.train_class_idx_[flat_indices]), flat_weights)

        proba = self.lam * avm_proba + (1 - self.lam) * knn_proba
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
