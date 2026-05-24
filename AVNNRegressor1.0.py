import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class AVNNRegressor(BaseEstimator, RegressorMixin):
    """
    Angular Vector Nearest Neighbor Regressor.

    Parameters
    ----------
    k : int, default=5
        Number of nearest neighbors.
    alpha : float, default=0.5
        Blend between angular (1) and Euclidean (0) distance.
    transform : str or callable, default='identity'
        Transform applied after min‑max normalisation.
        Built‑in: 'identity', 'tanh_ac', 'tan', 'neglog'.
    lam : float, default=1.0
        Blend between KNN prediction (1) and global mean (0).
    eps : float, default=1e-8
        Small constant for numerical stability.
    """
    def __init__(self, k=5, alpha=0.5, transform='identity', lam=1.0, eps=1e-8):
        self.k = k
        self.alpha = alpha
        self.transform = transform
        self.lam = lam
        self.eps = eps

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
        y = np.asarray(y).ravel()
        self.y_mean_ = np.mean(y)

        # Min‑max normalisation fit
        self.min_ = X.min(axis=0)
        self.range_ = np.maximum(X.max(axis=0) - self.min_, 1e-10)

        # Normalise and transform training data
        X_norm = (X - self.min_) / self.range_
        transform_fn = self._get_transform_fn()
        X_trans = transform_fn(X_norm)

        # Store for KNN
        self.X_norm_ = X_norm
        self.X_trans_ = X_trans
        self.y_ = y

        return self

    def _blended_distance(self, x_norm, x_trans):
        """Compute distances from a single point to all training points."""
        # Angular part
        theta_x = np.arccos(np.clip(x_norm, 0.0, 1.0))
        theta_train = np.arccos(np.clip(self.X_norm_, 0.0, 1.0))
        ang_diff = np.abs(theta_x - theta_train)          # (n_train, n_features)
        ang_dist = ang_diff.mean(axis=1)                  # (n_train,)

        # Euclidean part
        euc_diff = x_trans - self.X_trans_                # (n_train, n_features)
        euc_dist = np.sqrt((euc_diff ** 2).sum(axis=1) + self.eps)
        # euc_dist = np.sum(np.abs(euc_diff), axis=1) + self.eps

        return self.alpha * ang_dist + (1 - self.alpha) * euc_dist

    def predict(self, X):
        X = np.asarray(X)
        X_norm = (X - self.min_) / self.range_
        transform_fn = self._get_transform_fn()
        X_trans = transform_fn(X_norm)

        y_pred = np.zeros(len(X))
        for i, (xn, xt) in enumerate(zip(X_norm, X_trans)):
            dist = self._blended_distance(xn, xt)
            # Indices of k smallest distances
            idx = np.argpartition(dist, self.k)[:self.k]
            # Inverse distance weights
            w = 1.0 / (dist[idx] + self.eps)
            w = w / w.sum()
            y_knn = np.dot(w, self.y_[idx])
            y_pred[i] = self.lam * y_knn + (1 - self.lam) * self.y_mean_
        return y_pred
