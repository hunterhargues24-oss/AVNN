"""
AVNN Classifier
Best configuration: α=0.5, λ=0.7, k=5, transform='tanh_ac'
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# ----------------------------------------------------------------------
# Hybrid AVM + KNN Classifier
# ----------------------------------------------------------------------
class AVNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.5, lam=0.7, k=5, transform='tanh_ac', eps=1e-10):
        self.alpha = alpha
        self.lam = lam
        self.k = k
        self.transform = transform
        self.eps = eps
        # self._transform_fn = TRANSFORMS.get(transform, _identity)

    def _minmax_fit(self, X):
        self.mn_ = X.min(axis=0)
        self.rng_ = np.maximum(X.max(axis=0) - self.mn_, 1e-10)

    def _minmax_apply(self, X):
        return (X - self.mn_) / self.rng_

    def _prepare(self, X):
        X_norm = self._minmax_apply(X)
        transform_fn = self._get_transform_fn()
        X_trans = transform_fn(X_norm)
        return X_norm, X_trans

    def _tanh_ac(self, x, scale=1.6):
        x = np.clip(x, 0.0, 1.0)
        return np.tanh(np.arccos(x) * scale)

    def _identity(self, x):
        return x

    def _get_transform_fn(self):
        if callable(self.transform):
            return self.transform
        transforms = {
            'identity': self._identity,
            'tanh_ac': self._tanh_ac,
            'tan': lambda x: np.tan(np.clip(x, 0, 1) * np.pi / 2 * 0.97),
            'neglog': lambda x: -np.log(np.clip(x, 1e-10, 1.0))
        }
        return transforms[self.transform]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]

        self._minmax_fit(X)
        X_norm, X_trans = self._prepare(X)

        self.X_train_norm = X_norm
        self.X_train_trans = X_trans
        self.y_train = y

        self.centroids_norm_ = []
        self.centroids_trans_ = []
        for c in self.classes_:
            mask = (y == c)
            self.centroids_norm_.append(X_norm[mask].mean(axis=0))
            self.centroids_trans_.append(X_trans[mask].mean(axis=0))
        self.centroids_norm_ = np.array(self.centroids_norm_)
        self.centroids_trans_ = np.array(self.centroids_trans_)
        return self

    def _avm_proba(self, X_norm, X_trans):
        theta_x = np.arccos(np.clip(X_norm, 0.0, 1.0))
        theta_c = np.arccos(np.clip(self.centroids_norm_, 0.0, 1.0))
        ang_diff = np.abs(theta_x[:, None, :] - theta_c[None, :, :])
        ang_dist = ang_diff.mean(axis=2)

        euc_diff = X_trans[:, None, :] - self.centroids_trans_[None, :, :]
        euc_dist = np.sqrt((euc_diff ** 2).sum(axis=2) + self.eps)

        d = self.alpha * ang_dist + (1 - self.alpha) * euc_dist
        raw = 1.0 / (d + self.eps)
        return raw / raw.sum(axis=1, keepdims=True)

    def _knn_proba(self, X_norm, X_trans):
        theta_x = np.arccos(np.clip(X_norm, 0.0, 1.0))
        theta_train = np.arccos(np.clip(self.X_train_norm, 0.0, 1.0))
        ang_diff = np.abs(theta_x[:, None, :] - theta_train[None, :, :])
        ang_dist = ang_diff.mean(axis=2)

        euc_diff = X_trans[:, None, :] - self.X_train_trans[None, :, :]
        euc_dist = np.sqrt((euc_diff ** 2).sum(axis=2) + self.eps)

        d = self.alpha * ang_dist + (1 - self.alpha) * euc_dist

        N_test = X_norm.shape[0]
        K = len(self.classes_)
        knn_proba = np.zeros((N_test, K))
        for i in range(N_test):
            neigh_idx = np.argpartition(d[i], self.k)[:self.k]
            w = 1.0 / (d[i][neigh_idx] + self.eps)
            w = w / w.sum()
            for idx, weight in zip(neigh_idx, w):
                c = self.y_train[idx]
                class_idx = np.where(self.classes_ == c)[0][0]
                knn_proba[i, class_idx] += weight
        return knn_proba

    def predict_proba(self, X):
        X = np.asarray(X)
        X_norm, X_trans = self._prepare(X)
        avm_proba = self._avm_proba(X_norm, X_trans)
        knn_proba = self._knn_proba(X_norm, X_trans)
        proba = self.lam * avm_proba + (1 - self.lam) * knn_proba
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
