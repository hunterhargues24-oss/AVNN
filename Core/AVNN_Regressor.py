"""
LearningAVNN_Regressor
======================
Geometric tabular regressor — the continuous-output sibling of LearningAVNN.

Architecture Overview
---------------------
  Input → MinMax normalise [-1,1] → Six-branch combined vector (6F)
       → AVM: inverse-distance weighted mean of K×m learnable prototype values
       → KNN: FAISS inverse-distance weighted mean of neighbour target values
       → λ·AVM + (1-λ)·KNN  (λ auto-tuned post-training)

The AVM head computes:
  ŷ(x) = Σ_{k,j} w_{kj}(x) · v_{kj}  /  Σ_{k,j} w_{kj}(x)
  w_{kj}(x) = 1 / (d(x, μ_{kj}) / τ + ε)

Where μ_{kj} are learnable prototype positions (shape K×m×F) and
v_{kj} are learnable prototype target values (shape K×m), initialised
to the mean target of nearby training samples.

Six Branches (identical to LearningAVNN classifier)
----------------------------------------------------
  0  linear      x·√fw              direction + magnitude
  1  circular    √(1-x²)·√fw        symmetric extremeness
  2  boundary    √(‖x‖²-2x+1)       cross-feature coupling
  3  shape       z-score             within-sample relative profile
  4  quadratic   x²·√fw             nonlinear extremeness
  5  tanh_arccos tanh(arccos(x)·0.8)·√fw  monotone nonlinear, centre-amplified

Triangle Scoring (optional)
---------------------------
  Deviation cross-product for all F(F-1)/2 pairs — same as classifier.
  Proven +0.058 macro F1 on imbalanced classification; expected to help
  regression when target values correlate with feature ratio patterns.

Key Differences from Classifier
--------------------------------
  - Loss: Huber (robust to outliers) instead of NLL
  - Output: scalar ŷ instead of class probabilities
  - Prototype values v_{kj}: learnable scalars, not class labels
  - KNN: inverse-distance weighted mean of neighbour targets
  - No SupCon, ordinal EMD, class weights, or per-class tau
  - Regularisation: L2 penalty on prototype value spread
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class LearningAVNNRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, k=5, lr=1e-3, epochs=300, batch_size=256,
                 patience=60, val_fraction=0.15,
                 huber_delta=1.0,
                 weight_decay=1e-4, feat_weight_reg=0.05,
                 centroid_reg=0.1, centroid_sep=0.05,
                 proto_value_reg=0.1,
                 n_prototypes=1, intra_sep=0.05,
                 use_triangle=False, tri_weight=0.3,
                 ortho_reg=0.01,
                 mahalanobis=True, mahal_reg=1e-6, mahal_alpha=0.5,
                 lam_floor=0.5, entropy_lambda=False,
                 lam_confident=0.90, lam_uncertain=0.40,
                 ema_centroids=False, ema_beta=0.9,
                 use_ivf=True,
                 random_state=None, verbose=False):

        self.k                = k
        self.lr               = lr
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.patience         = patience
        self.val_fraction     = val_fraction
        self.huber_delta      = huber_delta
        self.weight_decay     = weight_decay
        self.feat_weight_reg  = feat_weight_reg
        self.centroid_reg     = centroid_reg
        self.centroid_sep     = centroid_sep
        self.proto_value_reg  = proto_value_reg
        self.n_prototypes     = n_prototypes
        self.intra_sep        = intra_sep
        self.use_triangle     = use_triangle
        self.tri_weight       = tri_weight
        self.ortho_reg        = ortho_reg
        self.mahalanobis      = mahalanobis
        self.mahal_reg        = mahal_reg
        self.mahal_alpha      = mahal_alpha
        self.lam_floor        = lam_floor
        self.entropy_lambda   = entropy_lambda
        self.lam_confident    = lam_confident
        self.lam_uncertain    = lam_uncertain
        self.ema_centroids    = ema_centroids
        self.ema_beta         = ema_beta
        self.use_ivf          = use_ivf
        self.random_state     = random_state
        self.verbose          = verbose

    # ── inner network ─────────────────────────────────────────────────────────

    class _Net(nn.Module):

        def __init__(self, centroids, proto_values, n_feat, m=1,
                     use_triangle=False, eps=1e-7):
            super().__init__()
            self.n_feat       = n_feat
            self.m            = m
            self.eps          = eps
            self.use_triangle = use_triangle
            self._sep_cache   = None

            # Prototype positions (K, m, F) and values (K, m)
            K = centroids.shape[0]
            self.K          = K
            self.centroids  = nn.Parameter(centroids.clone())
            self.register_buffer("centroid_anchor", centroids.clone())
            self.proto_vals = nn.Parameter(proto_values.clone())  # (K, m)

            # Global temperature — controls RBF sharpness
            self.log_tau = nn.Parameter(torch.tensor(-0.5))

            # Triangle scoring
            if use_triangle:
                n_pairs = n_feat * (n_feat - 1) // 2
                self.log_tau_tri = nn.Parameter(torch.tensor(-0.5))
                self.log_w_pairs = nn.Parameter(torch.zeros(n_pairs))

            # Six branch weights — independent sigmoid gates
            self.raw_w_lin  = nn.Parameter(torch.tensor(0.0))
            self.raw_w_cir  = nn.Parameter(torch.tensor(0.0))
            self.raw_w_b    = nn.Parameter(torch.tensor(0.5))  # boundary ↑
            self.raw_w_s    = nn.Parameter(torch.tensor(0.0))
            self.raw_w_sq   = nn.Parameter(torch.tensor(0.0))
            self.raw_w_tac  = nn.Parameter(torch.tensor(0.5))  # tac ↑
            self.raw_feat_w = nn.Parameter(torch.zeros(n_feat))

        def _branch_weights(self):
            raw = torch.stack([self.raw_w_lin, self.raw_w_cir, self.raw_w_b,
                               self.raw_w_s,   self.raw_w_sq,  self.raw_w_tac])
            g = torch.sigmoid(raw)
            return g / (g.sum() + 1e-10)

        def _feat_weights(self):
            return torch.sigmoid(self.raw_feat_w)

        @staticmethod
        def _boundary_dists(x, eps):
            norm_sq = (x ** 2).sum(dim=1, keepdim=True)
            return torch.sqrt((norm_sq - 2.0 * x + 1.0).clamp(min=eps))

        def _combined_features(self, x, wb, wf):
            eps  = self.eps
            sqwf = wf.sqrt()
            lin  = x * sqwf
            cir  = torch.sqrt(torch.clamp(1.0 - x*x, min=eps)) * sqwf
            bnd  = self._boundary_dists(x, eps)
            mu   = x.mean(1, keepdim=True)
            sd   = x.std(1,  keepdim=True).clamp(min=eps)
            shp  = (x - mu) / sd
            sqd  = x * x * sqwf
            a    = torch.acos(torch.clamp(x, -1+eps, 1-eps))
            tac  = torch.tanh(a * 0.8) * sqwf
            return torch.cat([wb[0].sqrt()*lin, wb[1].sqrt()*cir,
                               wb[2].sqrt()*bnd, wb[3].sqrt()*shp,
                               wb[4].sqrt()*sqd, wb[5].sqrt()*tac], dim=1)

        def _triangle_score(self, x, proto_flat):
            F, eps = self.n_feat, self.eps
            pa, pb = [], []
            for a in range(F):
                for b in range(a+1, F):
                    pa.append(a); pb.append(b)
            pa = torch.tensor(pa, device=x.device)
            pb = torch.tensor(pb, device=x.device)
            w  = torch.nn.functional.softplus(self.log_w_pairs)
            xd = x          - x.mean(1, keepdim=True)
            md = proto_flat - proto_flat.mean(1, keepdim=True)
            xa = xd[:, pa]; xb = xd[:, pb]
            ma = md[:, pa]; mb = md[:, pb]
            cross = (xa.unsqueeze(1)*mb.unsqueeze(0)
                   - xb.unsqueeze(1)*ma.unsqueeze(0))
            return torch.sqrt((w * cross**2).sum(-1) + eps)  # (N, K*m)

        def forward(self, x):
            """
            Returns (pred, feat_weights, embeddings).
            pred: (N,) weighted mean of prototype values.
            """
            tau  = torch.exp(self.log_tau)
            wb   = self._branch_weights()
            wf   = self._feat_weights()
            N    = x.shape[0]
            K, m = self.K, self.m

            fx         = self._combined_features(x, wb, wf)        # (N, D)
            proto_flat = self.centroids.reshape(K*m, self.n_feat)
            fy_flat    = self._combined_features(proto_flat, wb, wf)

            # Pairwise distances (N, K*m)
            sq_x  = (fx**2).sum(1, keepdim=True)
            sq_y  = (fy_flat**2).sum(1)
            dot   = fx @ fy_flat.T
            d_flat = torch.sqrt(torch.clamp(
                sq_x + sq_y[None,:] - 2.0*dot, min=self.eps))

            d = d_flat.reshape(N, K, m)

            # Triangle blend
            if self.use_triangle:
                tau_tri = torch.exp(self.log_tau_tri)
                t_flat  = self._triangle_score(x, proto_flat)
                t       = t_flat.reshape(N, K, m)
                inv_d   = (1.0 / (d / tau + self.eps)).sum(2)       # (N,K)
                inv_t   = (1.0 / (t / tau_tri + self.eps)).sum(2)
                inv_dn  = inv_d / inv_d.sum(-1,keepdim=True).clamp(min=self.eps)
                inv_tn  = inv_t / inv_t.sum(-1,keepdim=True).clamp(min=self.eps)
                tw      = getattr(self, '_tri_weight', 0.3)
                weights = (1-tw)*inv_dn + tw*inv_tn                 # (N,K)
            else:
                # Sum inverse-distances over prototypes per class
                weights = (1.0 / (d / tau + self.eps)).sum(2)       # (N,K)

            # Weighted mean of prototype values
            # proto_vals: (K,m) → mean over m → (K,)
            v    = self.proto_vals.mean(1)                           # (K,)
            w_n  = weights / weights.sum(-1, keepdim=True).clamp(min=self.eps)
            pred = (w_n * v.unsqueeze(0)).sum(-1)                   # (N,)

            return pred, wf, fx

    # ── normalisation ──────────────────────────────────────────────────────────

    def _norm_fit(self, X):
        self.mn_  = X.min(0)
        self.rng_ = np.maximum(X.max(0) - self.mn_, 1e-10)

    def _norm_apply(self, X):
        x = (X - self.mn_) / self.rng_
        return (-1.0 + 2.0 * x).astype(np.float32)

    # ── target scaling ─────────────────────────────────────────────────────────

    def _target_fit(self, y):
        self.y_mean_ = y.mean()
        self.y_std_  = max(y.std(), 1e-10)

    def _target_scale(self, y):
        return (y - self.y_mean_) / self.y_std_

    def _target_unscale(self, y):
        return y * self.y_std_ + self.y_mean_

    # ── loss ───────────────────────────────────────────────────────────────────

    def _loss(self, pred, targets, fw, embeddings=None):
        # Huber loss — robust to outlier target values
        huber = nn.functional.huber_loss(pred, targets,
                                         delta=self.huber_delta)

        # Feature weight regularisation — pull toward 1
        fw_reg = self.feat_weight_reg * ((fw - 1.0)**2).mean()

        # Centroid anchor — keep prototypes near class means
        c_anchor = self.centroid_reg * (
            (self.net_.centroids - self.net_.centroid_anchor)**2).mean()

        # Inter-class centroid separation
        K, m = self.net_.K, self.net_.m
        c_sep = torch.tensor(0.0)
        if self.net_._sep_cache is None:
            if K > 1:
                proto_flat = self.net_.centroids.reshape(K*m, self.net_.n_feat)
                c_dists    = torch.cdist(proto_flat, proto_flat, p=2)
                class_idx  = torch.arange(K).repeat_interleave(m)
                diff_mask  = (class_idx.unsqueeze(0) != class_idx.unsqueeze(1)).float()
                upper      = torch.triu(torch.ones_like(diff_mask), diagonal=1)
                mask       = diff_mask * upper
                if mask.sum() > 0:
                    c_sep = (-self.centroid_sep *
                             (c_dists * mask).sum() / mask.sum()).detach()
            self.net_._sep_cache = c_sep
        c_sep = self.net_._sep_cache

        # Prototype value regularisation — prevent value spread collapse/explosion
        v_reg = self.proto_value_reg * (self.net_.proto_vals**2).mean()

        # Ortho regularisation on branch outputs
        ortho = torch.tensor(0.0)
        if self.ortho_reg > 0.0 and embeddings is not None:
            n_branches = 6
            F_branch   = embeddings.shape[1] // n_branches
            chunks     = embeddings[:, :n_branches*F_branch].reshape(
                -1, n_branches, F_branch)
            normed     = nn.functional.normalize(chunks, dim=2)
            G          = torch.bmm(normed, normed.transpose(1,2)).mean(0)
            I          = torch.eye(n_branches, device=G.device)
            ortho      = self.ortho_reg * (G - I).pow(2).sum()

        return huber + fw_reg + c_anchor + c_sep + v_reg + ortho

    # ── val score ──────────────────────────────────────────────────────────────

    def _val_score(self, y_true, y_pred):
        """R² on unscaled targets."""
        return r2_score(self._target_unscale(y_true),
                        self._target_unscale(y_pred))

    # ── numpy combined vector (mirrors _Net exactly) ───────────────────────────

    @staticmethod
    def _build_make_combined(bw, fw):
        s = [float(np.sqrt(bw[i])) for i in range(6)]
        sqfw = np.sqrt(fw).astype(np.float32)

        def _make_combined(xn, batch_size=10_000):
            def _chunk(chunk):
                eps  = 1e-7
                lin  = chunk * sqfw
                cir  = np.sqrt(np.clip(1.0-chunk*chunk, eps, None)) * sqfw
                norm_sq = (chunk**2).sum(1, keepdims=True)
                bnd  = np.sqrt((norm_sq - 2.0*chunk + 1.0).clip(min=eps))
                mu   = chunk.mean(1, keepdims=True)
                sd   = chunk.std(1,  keepdims=True).clip(min=eps)
                shp  = (chunk - mu) / sd
                sqd  = chunk * chunk * sqfw
                a    = np.arccos(np.clip(chunk, -1+eps, 1-eps))
                tac  = np.tanh(a * 0.8) * sqfw
                return np.concatenate([
                    s[0]*lin, s[1]*cir, s[2]*bnd,
                    s[3]*shp, s[4]*sqd, s[5]*tac
                ], axis=1).astype(np.float32)
            if len(xn) <= batch_size:
                return _chunk(xn)
            return np.vstack([_chunk(xn[i:i+batch_size])
                               for i in range(0, len(xn), batch_size)])
        return _make_combined

    # ── AVM prediction ─────────────────────────────────────────────────────────

    def _avm_predict(self, combined):
        """Inverse-distance weighted mean of prototype values."""
        K  = self.net_.K
        m  = self.n_prototypes_
        N  = combined.shape[0]

        if self.cov_inv_ is not None:
            d_flat = np.zeros((N, K*m), dtype=np.float64)
            comb64 = combined.astype(np.float64)
            for km in range(K*m):
                k    = km // m
                diff = comb64 - self.centroids_combined_[km].astype(np.float64)
                tmp  = diff @ self.cov_inv_[k].astype(np.float64)
                d_flat[:, km] = np.sqrt(np.maximum((tmp*diff).sum(1), 0.0))
            d_flat = d_flat.astype(np.float32)
        else:
            a_sq   = (combined**2).sum(1, keepdims=True)
            b_sq   = (self.centroids_combined_**2).sum(1)
            ab     = combined @ self.centroids_combined_.T
            d_flat = np.sqrt(np.maximum(
                a_sq + b_sq[None,:] - 2.0*ab, 0.0)).astype(np.float32)

        # Weights: (N, K*m) → reshape (N, K, m) → sum over m → (N, K)
        w_flat = 1.0 / (d_flat + 1e-10)
        w      = w_flat.reshape(N, K, m).sum(2)              # (N, K)
        w_n    = w / w.sum(1, keepdims=True).clip(min=1e-10) # (N, K)

        # Prototype values: (K, m) → mean over m → (K,)
        v = self.proto_vals_np_                               # (K,)
        return (w_n * v[None,:]).sum(1)                      # (N,)

    # ── KNN prediction ─────────────────────────────────────────────────────────

    def _knn_predict(self, combined):
        """Inverse-distance weighted mean of k nearest neighbour targets."""
        dist, idx = self.index_.search(combined, self.k)
        w = 1.0 / (dist + 1e-10)
        w = w / w.sum(1, keepdims=True)
        return (w * self.train_targets_[idx]).sum(1)

    # ── fit ────────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_feat = X.shape[1]

        # Val split
        rng    = np.random.default_rng(self.random_state or 0)
        n_val  = max(1, int(len(X) * self.val_fraction))
        idx    = rng.permutation(len(X))
        va_idx, tr_idx = idx[:n_val], idx[n_val:]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        # Normalise X and y
        self._norm_fit(X_tr)
        X_tr_n = self._norm_apply(X_tr)
        X_va_n = self._norm_apply(X_va)

        self._target_fit(y_tr)
        y_tr_s = self._target_scale(y_tr).astype(np.float32)
        y_va_s = self._target_scale(y_va).astype(np.float32)

        m = max(1, int(self.n_prototypes))

        # Initialise prototype positions (K prototypes = n_prototypes clusters)
        # Use K-means-style init: divide target range into m buckets
        K = m  # for regression: K = n_prototypes (one centroid per region)
        # Simple percentile-based initialisation
        percentiles = np.linspace(0, 100, K+2)[1:-1]
        thresholds  = np.percentile(y_tr_s, percentiles)

        proto_init = np.zeros((K, 1, n_feat), dtype=np.float32)
        proto_vals = np.zeros((K, 1), dtype=np.float32)
        for i, t in enumerate(thresholds):
            # Assign samples to nearest threshold
            dists  = np.abs(y_tr_s - t)
            nearby = np.argsort(dists)[:max(5, len(X_tr)//K)]
            proto_init[i, 0] = X_tr_n[nearby].mean(0)
            proto_vals[i, 0] = float(t)

        centroids   = torch.tensor(proto_init)
        proto_v_t   = torch.tensor(proto_vals)

        self.net_ = self._Net(centroids, proto_v_t, n_feat, m=1,
                              use_triangle=self.use_triangle)
        self.net_._tri_weight = float(self.tri_weight)
        self.n_prototypes_    = 1

        tt = lambda a: torch.tensor(a, dtype=torch.float32)

        opt       = optim.Adam(self.net_.parameters(),
                               lr=self.lr, weight_decay=self.weight_decay)
        X_tr_t    = tt(X_tr_n)
        y_tr_t    = tt(y_tr_s)
        X_va_t    = tt(X_va_n)
        y_va_t    = tt(y_va_s)

        best_val, best_state, wait = -np.inf, None, 0

        for epoch in range(self.epochs):
            self.net_.train()
            self.net_._sep_cache = None
            perm = torch.randperm(len(X_tr_t))
            X_sh, y_sh = X_tr_t[perm], y_tr_t[perm]
            bs = self.batch_size

            for i in range(0, len(X_sh), bs):
                xb, yb = X_sh[i:i+bs], y_sh[i:i+bs]
                opt.zero_grad()
                pred, fw, emb = self.net_(xb)
                self._loss(pred, yb, fw, embeddings=emb).backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()

            # EMA centroid update
            if self.ema_centroids:
                self.net_.eval()
                with torch.no_grad():
                    for ki in range(K):
                        # For regression: EMA based on target proximity
                        dists_k = np.abs(y_tr_s - proto_vals[ki, 0])
                        near_k  = dists_k < (dists_k.std() + 1e-10)
                        if near_k.sum() > 0:
                            mean_k = tt(X_tr_n[near_k].mean(0))
                            self.net_.centroids.data[ki, 0] = (
                                self.ema_beta * self.net_.centroids.data[ki, 0]
                                + (1 - self.ema_beta) * mean_k)

            # Validation
            if (epoch + 1) % 5 == 0:
                self.net_.eval()
                with torch.no_grad():
                    pred_va, _, _ = self.net_(X_va_t)
                    val_r2 = self._val_score(
                        y_va_s, pred_va.numpy())
                if val_r2 > best_val:
                    best_val   = val_r2
                    best_state = {k: v.clone()
                                  for k, v in self.net_.state_dict().items()}
                    wait = 0
                else:
                    wait += 5
                    if wait >= self.patience:
                        if self.verbose:
                            print(f"  early stopping at epoch {epoch+1}")
                        break
                if self.verbose:
                    bw = self.net_._branch_weights().detach().numpy()
                    print(f"  epoch {epoch+1:3d}  val_r2={val_r2:.4f}"
                          f"  branches=[lin:{bw[0]:.2f} cir:{bw[1]:.2f}"
                          f" bnd:{bw[2]:.2f} shp:{bw[3]:.2f}"
                          f" sq:{bw[4]:.2f} tac:{bw[5]:.2f}]")

        if best_state:
            self.net_.load_state_dict(best_state)

        # Extract learned params for numpy inference
        self.net_.eval()
        with torch.no_grad():
            bw = self.net_._branch_weights().numpy()
            fw = self.net_._feat_weights().numpy()

        _make_combined = self._build_make_combined(bw, fw)

        raw_c    = self.net_.centroids.detach().numpy()       # (K, 1, F)
        raw_c_f  = raw_c.reshape(K, n_feat)
        self.centroids_combined_ = _make_combined(raw_c_f)   # (K, D)
        self.proto_vals_np_      = (
            self.net_.proto_vals.detach().numpy().mean(1))    # (K,)

        # Mahalanobis covariance
        self.cov_inv_ = None
        if self.mahalanobis:
            X_tr_c = _make_combined(X_tr_n)
            D      = X_tr_c.shape[1]
            S_pool = np.zeros((D, D), dtype=np.float64)
            for ki in range(K):
                dists_k = np.abs(y_tr_s - proto_vals[ki, 0])
                near_k  = dists_k < (dists_k.std() + 1e-10)
                pts     = X_tr_c[near_k].astype(np.float64)
                if len(pts) < 2: continue
                c = pts - pts.mean(0)
                S_pool += c.T @ c
            n_total = len(X_tr_c)
            S_pool /= max(n_total - K, 1)

            cov_inv = []
            for ki in range(K):
                dists_k = np.abs(y_tr_s - proto_vals[ki, 0])
                near_k  = dists_k < (dists_k.std() + 1e-10)
                pts     = X_tr_c[near_k].astype(np.float64)
                if len(pts) >= 2:
                    c   = pts - pts.mean(0, keepdims=True)
                    S_k = (c.T @ c) / max(len(pts)-1, 1)
                else:
                    S_k = S_pool.copy()
                S_rda = ((1-self.mahal_alpha)*S_k
                         + self.mahal_alpha*S_pool
                         + self.mahal_reg * np.eye(D))
                try:
                    L = np.linalg.cholesky(S_rda)
                    cov_inv.append(
                        np.linalg.solve(L.T, np.linalg.solve(L, np.eye(D))))
                except np.linalg.LinAlgError:
                    cov_inv.append(np.linalg.pinv(S_rda))
            self.cov_inv_ = np.stack(cov_inv).astype(np.float64)
            if self.verbose:
                print(f"  Mahalanobis: full RDA (alpha={self.mahal_alpha} "
                      f"ridge={self.mahal_reg} D={D})")

        # FAISS index
        self.index_         = None
        self.train_targets_ = None
        if _HAS_FAISS:
            X_tr_c = _make_combined(X_tr_n)
            N_build, D = X_tr_c.shape
            self.train_targets_ = y_tr_s.copy()
            if self.use_ivf and N_build >= 78:
                nlist     = min(1000, max(1, N_build // 39))
                quantizer = faiss.IndexFlatL2(D)
                index     = faiss.IndexIVFFlat(quantizer, D, nlist,
                                               faiss.METRIC_L2)
                index.train(X_tr_c)
                index.nprobe = min(10, nlist)
            else:
                index = faiss.IndexFlatL2(D)
            index.add(X_tr_c)
            self.index_ = index

        # Lambda search
        if self.index_ is not None:
            X_va_c  = _make_combined(X_va_n)
            avm_val = self._avm_predict(X_va_c)
            knn_val = self._knn_predict(X_va_c)
            best_r2, best_lam = -np.inf, self.lam_floor
            for lam in np.arange(self.lam_floor, 1.01, 0.05):
                lam  = float(np.round(lam, 6))
                pred = lam*avm_val + (1-lam)*knn_val
                r2   = self._val_score(y_va_s, pred)
                if r2 > best_r2:
                    best_r2, best_lam = r2, lam
            self.lambda_ = best_lam
        else:
            self.lambda_ = 1.0

        self._make_combined = _make_combined
        self.n_features_    = n_feat

        if self.verbose:
            print(f"  lambda search: best={self.lambda_:.2f}"
                  f"  val_r2={best_val:.4f}")

        return self

    # ── predict ────────────────────────────────────────────────────────────────

    def predict(self, X):
        check_is_fitted(self, 'centroids_combined_')
        X        = np.asarray(X, dtype=np.float32)
        Xn       = self._norm_apply(X)
        combined = self._make_combined(Xn)
        avm      = self._avm_predict(combined)

        if self.index_ is None or self.lambda_ >= 1.0:
            return self._target_unscale(avm)

        knn = self._knn_predict(combined)

        if self.entropy_lambda:
            # Use prediction variance as uncertainty proxy
            # High variance between AVM and KNN → low lambda
            diff = np.abs(avm - knn) / (np.abs(avm) + np.abs(knn) + 1e-10)
            lam  = (self.lam_confident
                    - (self.lam_confident - self.lam_uncertain) * diff)
            lam  = lam.clip(self.lam_uncertain, self.lam_confident)
            pred = lam * avm + (1 - lam) * knn
        else:
            pred = self.lambda_ * avm + (1 - self.lambda_) * knn

        return self._target_unscale(pred)

    # ── summary ────────────────────────────────────────────────────────────────

    def summary(self, feature_names=None):
        check_is_fitted(self, 'centroids_combined_')
        bw = self.net_._branch_weights().detach().numpy()
        lam = getattr(self, 'lambda_', self.lam_floor)
        lam_str = f"{lam:.3f}" if lam is not None else "[entropy]"
        lines = [
            f"  lambda={lam_str}  entropy_lambda={self.entropy_lambda}",
            f"  n_prototypes={self.n_prototypes}  "
            f"mahalanobis={'full' if self.mahalanobis else 'off'}"
            f"  alpha={self.mahal_alpha}",
            f"  triangle={self.use_triangle}  "
            f"huber_delta={self.huber_delta}",
            f"  Branch weights: lin={bw[0]:.3f} cir={bw[1]:.3f}"
            f" bnd={bw[2]:.3f} shp={bw[3]:.3f}"
            f" sq={bw[4]:.3f} tac={bw[5]:.3f}",
        ]
        if feature_names is not None:
            fw = self.net_._feat_weights().detach().numpy()
            top = np.argsort(fw)[::-1][:5]
            lines.append("  Top feature weights:")
            for i in top:
                lines.append(f"    {list(feature_names)[i]:<28} {fw[i]:.4f}")
        lines.append(f"  Prototype values: "
                     + "  ".join(f"p{i}={v:.3f}"
                                 for i, v in enumerate(self.proto_vals_np_)))
        return "\n".join(lines)
