"""
FastTriBranchAVNN
==================

Static three-branch AVNN classifier with FAISS-accelerated KNN.
No training required — pass branch weights directly or use defaults.

The deploy-time counterpart to BranchAdaptiveAVNN: train the learnable
version once to find good weights, then pass them here for fast inference.

Three branches
--------------
  1. Angular  — axis-separable arccos differences, per-feature weighted
  2. Euclidean — L2 on tanh_ac-transformed features
  3. Shape    — L2 between per-sample z-score normalised vectors

Train == inference (no mismatch)
---------------------------------
Everything — the FAISS index, AVM centroids, and query scoring — operates
on the SAME combined vector:

    combined = concat(√w_ang * √fw * ang,  √w_euc * euc,  √w_shape * shape)

Squared L2 distance in this space equals the branch-weighted sum of
squared per-branch distances. AVM centroids are the class means of this
combined vector. FAISS indexes the same combined vectors. There is one
distance metric, used identically everywhere.

This is a deliberate squared-L2 approximation of the true weighted-L2
blended distance. Neighbour ordering is preserved under monotone transforms
so the KNN and AVM branches remain consistent with each other.

Parameters
----------
lam : float, default=0.7
    AVM vs KNN blend. 1.0 = pure AVM (no FAISS required or built).
k : int, default=5
    Nearest neighbours for KNN branch.
w_ang, w_euc, w_shape : float
    Branch weights. Normalised internally to sum to 1.
feat_weights : array-like or None
    Per-feature angular weights (e.g. from BranchAdaptiveAVNN.angular_feat_weights_).
    None = uniform weighting.
transform : 'tanh_ac' | 'identity'
    Euclidean branch transform applied after normalisation.
norm_range : '[-1,1]' | '[0,1]'
    Normalisation range. '[-1,1]' recommended (full semicircle for arccos).
use_ivf : bool, default=True
    IVF approximate index (fast) vs flat exact index.
    Automatically falls back to flat for small datasets (< 78 samples).
nlist : int, default=1000
    IVF Voronoi cell count. Capped internally at n_train // 39.
nprobe : int, default=10
    Cells to search per query. Higher = more accurate, slower.
eps : float, default=1e-10
    Numerical stability constant.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class FastTriBranchAVNN(BaseEstimator, ClassifierMixin):

    def __init__(self, lam=0.7, k=5,
                 w_ang=0.34, w_euc=0.33, w_shape=0.33,
                 feat_weights=None, transform='tanh_ac',
                 norm_range='[-1,1]', eps=1e-10,
                 use_ivf=True, nlist=1000, nprobe=10):
        self.lam          = lam
        self.k            = k
        self.w_ang        = w_ang
        self.w_euc        = w_euc
        self.w_shape      = w_shape
        self.feat_weights = feat_weights
        self.transform    = transform
        self.norm_range   = norm_range
        self.eps          = eps
        self.use_ivf      = use_ivf
        self.nlist        = nlist
        self.nprobe       = nprobe

    # ── preprocessing ─────────────────────────────────────────────────────────

    def _norm_fit(self, X):
        self.mn_  = X.min(0)
        self.rng_ = np.maximum(X.max(0) - self.mn_, 1e-10)

    def _norm_apply(self, X):
        x = (X - self.mn_) / self.rng_
        if self.norm_range == '[-1,1]':
            x = -1.0 + 2.0 * x
        return x.astype(np.float32)

    def _trans_apply(self, x_norm):
        if self.transform == 'identity':
            return x_norm.astype(np.float32)
        # tanh_ac
        if self.norm_range == '[-1,1]':
            x = np.clip(x_norm, -1.0 + 1e-7, 1.0 - 1e-7)
            return np.tanh(np.arccos(x) * 0.8).astype(np.float32)
        x = np.clip(x_norm, 0.0, 1.0)
        return np.tanh(np.arccos(x) * 1.6).astype(np.float32)

    def _arccos_clip(self, x):
        lo = -1.0 + self.eps if self.norm_range == '[-1,1]' else self.eps
        return np.arccos(np.clip(x, lo, 1.0 - self.eps)).astype(np.float32)

    @staticmethod
    def _shape(x_norm, eps=1e-7):
        mu = x_norm.mean(1, keepdims=True)
        sd = x_norm.std(1,  keepdims=True).clip(min=eps)
        return ((x_norm - mu) / sd).astype(np.float32)

    # ── combined vector ───────────────────────────────────────────────────────

    def _combined(self, x_norm, x_trans):
        """
        Single combined vector used for BOTH centroids and queries.

        Scaling: each branch component is multiplied by sqrt(branch_weight)
        so that squared L2 in the concatenated space equals the
        branch-weighted sum of squared per-branch distances.

        Per-feature angular weights are folded in as sqrt(feat_weight),
        preserving their relative importance in the angular distance.
        """
        ang   = self._arccos_clip(x_norm) * self._sqrt_feat_w_  # (N, F)
        euc   = x_trans                                          # (N, F)
        shape = self._shape(x_norm)                             # (N, F)

        return np.concatenate([
            self._s_ang_   * ang,
            self._s_euc_   * euc,
            self._s_shape_ * shape,
        ], axis=1).astype(np.float32)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.classes_    = np.unique(y)
        self.n_features_ = X.shape[1]

        # Normalise branch weights, store sqrt scales
        total         = self.w_ang + self.w_euc + self.w_shape
        self._s_ang_   = np.float32(np.sqrt(self.w_ang   / total))
        self._s_euc_   = np.float32(np.sqrt(self.w_euc   / total))
        self._s_shape_ = np.float32(np.sqrt(self.w_shape / total))

        # Per-feature angular weights
        if self.feat_weights is None:
            fw = np.ones(self.n_features_, dtype=np.float32)
        else:
            fw = np.asarray(self.feat_weights, dtype=np.float32)
            if len(fw) != self.n_features_:
                raise ValueError(
                    f"feat_weights length {len(fw)} != n_features {self.n_features_}")
        self._sqrt_feat_w_ = np.sqrt(np.maximum(fw, 0.0)).astype(np.float32)

        # Normalise and build combined vectors
        self._norm_fit(X)
        Xn       = self._norm_apply(X)
        Xt       = self._trans_apply(Xn)
        combined = self._combined(Xn, Xt)

        # Class index mapping for fast voting
        self.class_to_idx_    = {c: i for i, c in enumerate(self.classes_)}
        self.train_class_idx_ = np.array(
            [self.class_to_idx_[c] for c in y], dtype=np.int64)

        # AVM centroids in combined space — same metric as FAISS
        self.centroids_ = np.stack([
            combined[y == c].mean(0) for c in self.classes_
        ]).astype(np.float32)

        # FAISS index — only built when KNN branch is active
        if self.lam < 1.0:
            if not _HAS_FAISS:
                raise ImportError(
                    "faiss not installed. Install faiss-cpu, or set lam=1.0 "
                    "for pure AVM (no FAISS needed).")
            dim = combined.shape[1]
            if self.use_ivf and combined.shape[0] >= 78:
                nlist = max(1, min(self.nlist, combined.shape[0] // 39))
                quantizer = faiss.IndexFlatL2(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                           faiss.METRIC_L2)
                index.train(combined)
                index.nprobe = self.nprobe
            else:
                index = faiss.IndexFlatL2(dim)
            index.add(combined)
            self.index_ = index

        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _avm_proba(self, combined):
        """
        Inverse-L2 to combined-space centroids. (N, K)

        Uses ||a-b||² = ||a||² - 2a·b + ||b||² to avoid the
        (N, K, F) intermediate that becomes ~400MB at N=434k.
        """
        a_sq = (combined   ** 2).sum(1, keepdims=True)   # (N, 1)
        b_sq = (self.centroids_ ** 2).sum(1)             # (K,)
        ab   = combined @ self.centroids_.T               # (N, K)
        d    = np.sqrt(np.maximum(a_sq + b_sq[None, :] - 2.0 * ab,
                                   0.0)).astype(np.float32)
        raw  = 1.0 / (d + self.eps)
        return (raw / raw.sum(1, keepdims=True)).astype(np.float32)

    def _knn_proba(self, combined):
        """FAISS KNN inverse-distance vote. (N, K)"""
        dist, idx = self.index_.search(combined, self.k)
        w = 1.0 / (dist + self.eps)
        w = (w / w.sum(1, keepdims=True)).astype(np.float32)

        n_test = combined.shape[0]
        K      = len(self.classes_)
        proba  = np.zeros((n_test, K), dtype=np.float32)
        t_idx  = np.repeat(np.arange(n_test), self.k)
        np.add.at(proba,
                  (t_idx, self.train_class_idx_[idx.ravel()]),
                  w.ravel())
        return proba

    def predict_proba(self, X):
        check_is_fitted(self, 'centroids_')
        X        = np.asarray(X, dtype=np.float32)
        Xn       = self._norm_apply(X)
        Xt       = self._trans_apply(Xn)
        combined = self._combined(Xn, Xt)

        avm = self._avm_proba(combined)
        if self.lam >= 1.0:
            return avm
        knn = self._knn_proba(combined)
        return self.lam * avm + (1.0 - self.lam) * knn

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def summary(self, feature_names=None):
        check_is_fitted(self, 'centroids_')
        names = (list(feature_names) if feature_names is not None
                 else [f"f{i}" for i in range(self.n_features_)])
        total = self.w_ang + self.w_euc + self.w_shape
        lines = [
            f"  lam={self.lam:.3f}  k={self.k}",
            f"  Branch weights: angular={self.w_ang/total:.3f}  "
            f"euclidean={self.w_euc/total:.3f}  shape={self.w_shape/total:.3f}",
        ]
        if self.feat_weights is not None:
            lines.append("  Angular per-feature weights:")
            fw    = np.asarray(self.feat_weights)
            order = np.argsort(fw)[::-1]
            for i in order:
                lines.append(f"    {names[i]:<26} {fw[i]:.4f}")
        return "\n".join(lines)
