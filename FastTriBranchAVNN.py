import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class FastTriBranchAVNN(BaseEstimator, ClassifierMixin):
    """
    Fast three‑branch AVNN using FAISS for approximate KNN, but with
    correct blended distance (weighted sum, not weighted Euclidean).
    """

    def __init__(self, lam=0.7, k=5,
                 w_ang=0.333, w_euc=0.333, w_shape=0.334,
                 feat_weights=None, transform='tanh_ac',
                 norm_range='[-1,1]', eps=1e-10,
                 use_ivf=True, nlist=1000, nprobe=10):
        self.lam = lam
        self.k = k
        self.w_ang = w_ang
        self.w_euc = w_euc
        self.w_shape = w_shape
        self.feat_weights = feat_weights
        self.transform = transform
        self.norm_range = norm_range
        self.eps = eps
        self.use_ivf = use_ivf
        self.nlist = nlist
        self.nprobe = nprobe

    # ---------- preprocessing ----------
    def _norm_fit(self, X):
        self.mn_ = X.min(0)
        self.rng_ = np.maximum(X.max(0) - self.mn_, 1e-10)

    def _norm_apply(self, X):
        x = (X - self.mn_) / self.rng_
        if self.norm_range == '[-1,1]':
            x = -1.0 + 2.0 * x
        return x.astype(np.float32)

    def _tanh_ac(self, x):
        if self.norm_range == '[-1,1]':
            x = np.clip(x, -1.0 + 1e-7, 1.0 - 1e-7)
            return np.tanh(np.arccos(x) * 0.8).astype(np.float32)
        x = np.clip(x, 0.0, 1.0)
        return np.tanh(np.arccos(x) * 1.6).astype(np.float32)

    def _trans_apply(self, x_norm):
        if self.transform == 'identity':
            return x_norm.astype(np.float32)
        return self._tanh_ac(x_norm)

    def _arccos_clip(self, x):
        lo = -1.0 + self.eps if self.norm_range == '[-1,1]' else self.eps
        return np.arccos(np.clip(x, lo, 1.0 - self.eps))

    # ---------- distance components ----------
    def _angular_dist(self, xn_a, xn_b):
        """Weighted axis‑separable angular distance."""
        w_feat = self._sqrt_feat_w_ ** 2   # original weights (before sqrt)
        th_a = self._arccos_clip(xn_a)
        th_b = self._arccos_clip(xn_b)
        # (N, M) matrix of weighted mean absolute differences
        diff = np.abs(th_a[:, None, :] - th_b[None, :, :])  # (N, M, F)
        ang = np.dot(diff.reshape(-1, self.n_features_), w_feat).reshape(diff.shape[:2])
        return ang

    def _euclidean_dist(self, xt_a, xt_b):
        diff = xt_a[:, None, :] - xt_b[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2) + self.eps)

    def _shape_dist(self, xn_a, xn_b):
        # per‑sample z‑score
        mu_a = xn_a.mean(axis=1, keepdims=True)
        sd_a = xn_a.std(axis=1, keepdims=True).clip(min=self.eps)
        mu_b = xn_b.mean(axis=1, keepdims=True)
        sd_b = xn_b.std(axis=1, keepdims=True).clip(min=self.eps)
        za = (xn_a - mu_a) / sd_a
        zb = (xn_b - mu_b) / sd_b
        diff = za[:, None, :] - zb[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2) + self.eps)

    def _blended_dist(self, xn_a, xt_a, xn_b, xt_b):
        """True blended distance = weighted sum of branch distances."""
        d_ang = self._angular_dist(xn_a, xn_b)
        d_euc = self._euclidean_dist(xt_a, xt_b)
        d_shape = self._shape_dist(xn_a, xn_b)
        return self.w_ang * d_ang + self.w_euc * d_euc + self.w_shape * d_shape

    # ---------- combined vector for FAISS (approximate) ----------
    def _combined_vector(self, xn, xt):
        """
        Build a vector such that Euclidean distance approximates
        the true blended distance (for neighbour search only).
        Uses sqrt(branch weights) and no angular feature weights.
        """
        ang = self._arccos_clip(xn)
        euc = xt
        shape = self._shape(xn)   # we need a shape function without feature weights
        scale_ang = np.sqrt(self.w_ang)
        scale_euc = np.sqrt(self.w_euc)
        scale_shape = np.sqrt(self.w_shape)
        # Note: angular feature weights are omitted here – they would break
        # the Euclidean approximation. This is acceptable for approximate search.
        return np.concatenate([scale_ang * ang, scale_euc * euc, scale_shape * shape], axis=1)

    @staticmethod
    def _shape(xn):
        mu = xn.mean(1, keepdims=True)
        sd = xn.std(1, keepdims=True).clip(min=1e-7)
        return (xn - mu) / sd

    # ---------- fit ----------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        K = len(self.classes_)

        # Normalise branch weights
        total = self.w_ang + self.w_euc + self.w_shape
        self.w_ang /= total
        self.w_euc /= total
        self.w_shape /= total

        # Angular per‑feature weights
        if self.feat_weights is None:
            fw = np.ones(self.n_features_, dtype=np.float32)
        else:
            fw = np.asarray(self.feat_weights, dtype=np.float32)
            if len(fw) != self.n_features_:
                raise ValueError(f"feat_weights length {len(fw)} != n_features {self.n_features_}")
        # Store original weights for distance computation
        self._feat_weights_ = fw
        # Also store sqrt for scaling in angular distance? Actually we need them as is.
        self._sqrt_feat_w_ = np.sqrt(np.maximum(fw, 0.0))

        # Preprocess training data
        self._norm_fit(X)
        Xn = self._norm_apply(X)
        Xt = self._trans_apply(Xn)
        # Compute true blended distances to centroids (for AVM branch later) –
        # we store centroids in original space, not combined.
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        self.y_train_ = y
        self.X_train_norm_ = Xn
        self.X_train_trans_ = Xt

        # Centroids in original normalised and transformed spaces
        self.centroids_norm_ = np.stack([Xn[y == c].mean(0) for c in self.classes_])
        self.centroids_trans_ = np.stack([Xt[y == c].mean(0) for c in self.classes_])

        # Build FAISS index for KNN (using approximate combined vector)
        if self.lam < 1.0:
            if not _HAS_FAISS:
                raise ImportError("faiss not installed. Install faiss-cpu or set lam=1.0.")
            combined = self._combined_vector(Xn, Xt)
            dim = combined.shape[1]
            if self.use_ivf and combined.shape[0] >= 78:
                nlist = max(1, min(self.nlist, combined.shape[0] // 39))
                quantizer = faiss.IndexFlatL2(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
                index.train(combined)
                index.nprobe = self.nprobe
            else:
                index = faiss.IndexFlatL2(dim)
            index.add(combined)
            self.index_ = index

        return self

    # ---------- predict ----------
    def _avm_proba(self, Xn, Xt):
        """True blended distance to centroids."""
        d = self._blended_dist(Xn, Xt, self.centroids_norm_, self.centroids_trans_)
        raw = 1.0 / (d + self.eps)
        return raw / raw.sum(axis=1, keepdims=True)

    def _knn_proba(self, Xn, Xt):
        """Use FAISS to find neighbours, then recompute true blended distance for weighting."""
        # Build combined vector for queries
        combined_q = self._combined_vector(Xn, Xt)
        # Retrieve nearest neighbours (indices)
        distances_approx, indices = self.index_.search(combined_q, self.k)

        # For each test point, compute true distances to those k neighbours
        n_test = Xn.shape[0]
        K = len(self.classes_)
        knn_proba = np.zeros((n_test, K), dtype=np.float32)

        # Pre‑extract neighbour normed and transformed arrays for all queries at once?
        # We'll loop over test points because k is small.
        for i in range(n_test):
            idx = indices[i]
            # Fetch training points' norm and trans
            xn_neigh = self.X_train_norm_[idx]
            xt_neigh = self.X_train_trans_[idx]
            # True blended distances
            d_true = self._blended_dist(Xn[i:i+1], Xt[i:i+1], xn_neigh, xt_neigh).ravel()
            w = 1.0 / (d_true + self.eps)
            w /= w.sum()
            for j, weight in zip(idx, w):
                c = self.y_train_[j]
                class_idx = self.class_to_idx_[c]
                knn_proba[i, class_idx] += weight
        return knn_proba

    def predict_proba(self, X):
        check_is_fitted(self, 'centroids_norm_')
        X = np.asarray(X, dtype=np.float32)
        Xn = self._norm_apply(X)
        Xt = self._trans_apply(Xn)

        avm = self._avm_proba(Xn, Xt)
        if self.lam >= 1.0:
            return avm
        knn = self._knn_proba(Xn, Xt)
        return self.lam * avm + (1 - self.lam) * knn

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def summary(self, feature_names=None):
        names = feature_names or [f"f{i}" for i in range(self.n_features_)]
        lines = [f"lambda={self.lam:.4f}, k={self.k}"]
        lines.append(f"Branch weights: angular={self.w_ang:.4f}, "
                     f"euclidean={self.w_euc:.4f}, shape={self.w_shape:.4f}")
        lines.append("Angular per‑feature weights:")
        for n, w in zip(names, self._feat_weights_):
            lines.append(f"  {n:<24} {w:.4f}")
        return "\n".join(lines)
