"""
AutoLambdaAVNN
==============

Four-branch AVNN classifier with automatic lambda tuning and gravity.

Architecture
------------
Four distance branches in [-1,1] normalised space:

  phi_dir  = arccos(x_f)        direction-aware angle      [0, π]
  phi_mag  = arccos(|x_f|)      magnitude angle (symmetric) [0, π/2]
  euc      = tanh(arccos(x)*0.8) nonlinear transform
  shape    = z-score(x)          per-sample profile

Combined vector:
    [√w_d * √fw * phi_dir,  √w_m * √fw * phi_mag,
     √w_e * √fw * euc,      √w_s * shape]

Squared L2 in this space = branch-weighted feature-weighted distance.
One metric used identically in training (PyTorch) and inference (numpy/FAISS).

Training strategy
-----------------
- Pure AVM (centroid-only) during training — O(batch × K × F), fast.
- Gravity applied inside forward() so gradients see the full pipeline.
- Class-weighted NLL with halved cap when gravity > 0 (prevents double-correction).
- Feature weight L2 penalty anchors weights near uniform — avoids overfit on small data.
- No WARMUP epochs before patience counting starts.
- Post-training lambda grid search on validation set using full AVM+KNN+gravity.

Parameters
----------
k               KNN neighbours (default 5)
lr              Adam learning rate (default 1e-3)
epochs          Max training epochs (default 300)
batch_size      Mini-batch size (default 64)
patience        Early stopping patience on val macro F1 (default 60)
val_fraction    Validation holdout (default 0.15)
label_smoothing NLL label smoothing (default 0.05)
weight_cap      Max class weight in NLL (default 1.5; halved when gravity>0)
weight_decay    Adam L2 regularisation (default 1e-4)
feat_weight_reg L2 penalty anchoring feature weights near uniform (default 0.05)
gravity         Inference-time minority class boost (default 0.0)
gravity_cap     Symmetric weight clip: weights stay in [1/cap, cap] (default 1.5)
lam_init        Starting lambda if auto_lambda=False (default 0.7)
auto_lambda     Search for best lambda post-training (default True)
lam_search_step Grid step for lambda search (default 0.05)
use_ivf         FAISS IVF approximate index (default True)
nlist           IVF cluster count (default 1000, capped at n//39)
nprobe          IVF cells searched per query (default 10)
random_state    Reproducibility seed (default None)
verbose         Print training progress (default False)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import f1_score

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class AutoLambdaAVNN(BaseEstimator, ClassifierMixin):

    def __init__(self, k=5, lr=1e-3, epochs=300, batch_size=256,
                 patience=60, val_fraction=0.15, label_smoothing=0.05,
                 weight_cap=1.5, weight_decay=1e-4, feat_weight_reg=0.05,
                 gravity=0.0, gravity_cap=1.5,
                 lam_init=0.7, auto_lambda=True, lam_search_step=0.05,
                 ordinal=False, ordinal_weight=0.5,
                 per_class_tau=False,
                 val_acc_weight=0.33, val_macro_weight=0.34,
                 val_weighted_weight=0.33,
                 use_ivf=True, nlist=1000, nprobe=10,
                 random_state=None, verbose=False):
        self.k                   = k
        self.lr                  = lr
        self.epochs              = epochs
        self.batch_size          = batch_size
        self.patience            = patience
        self.val_fraction        = val_fraction
        self.label_smoothing     = label_smoothing
        self.weight_cap          = weight_cap
        self.weight_decay        = weight_decay
        self.feat_weight_reg     = feat_weight_reg
        self.gravity             = gravity
        self.gravity_cap         = gravity_cap
        self.lam_init            = lam_init
        self.auto_lambda         = auto_lambda
        self.lam_search_step     = lam_search_step
        self.ordinal             = ordinal
        self.ordinal_weight      = ordinal_weight
        self.per_class_tau       = per_class_tau
        self.val_acc_weight      = val_acc_weight
        self.val_macro_weight    = val_macro_weight
        self.val_weighted_weight = val_weighted_weight
        self.use_ivf             = use_ivf
        self.nlist               = nlist
        self.nprobe              = nprobe
        self.random_state        = random_state
        self.verbose             = verbose

    # ── preprocessing ─────────────────────────────────────────────────────────

    def _norm_fit(self, X):
        self.mn_  = X.min(0)
        self.rng_ = np.maximum(X.max(0) - self.mn_, 1e-10)

    def _norm_apply(self, X):
        return (-1.0 + 2.0 * (X - self.mn_) / self.rng_).astype(np.float32)

    # ── inner PyTorch model ───────────────────────────────────────────────────

    class _Net(nn.Module):

        def __init__(self, centroids, K, n_feat, per_class_tau=False, eps=1e-7):
            super().__init__()
            self.K, self.n_feat, self.eps = K, n_feat, eps
            self.per_class_tau = per_class_tau
            self._centroid_cache = None

            self.register_buffer("centroids", centroids)
            self.register_buffer("grav_w", torch.ones(K))

            # Per-class tau: each class centroid gets its own scoring temperature.
            # Minority class centroids are noisy (few samples) — they learn higher
            # tau (softer scoring). Majority class centroids are stable — they
            # learn lower tau (sharper scoring).
            if per_class_tau:
                self.log_tau = nn.Parameter(torch.full((K,), -0.5))
            else:
                self.log_tau = nn.Parameter(torch.tensor(-0.5))

            self.raw_w_d    = nn.Parameter(torch.tensor(0.0))
            self.raw_w_m    = nn.Parameter(torch.tensor(0.0))
            self.raw_w_e    = nn.Parameter(torch.tensor(0.0))
            self.raw_w_s    = nn.Parameter(torch.tensor(0.0))
            self.raw_feat_w = nn.Parameter(torch.zeros(n_feat))

        def _branch_weights(self):
            return torch.softmax(torch.stack([
                self.raw_w_d, self.raw_w_m,
                self.raw_w_e, self.raw_w_s]), dim=0)

        def _feat_weights(self):
            w = torch.softmax(self.raw_feat_w, dim=0)
            return w * self.n_feat

        def _combined_features(self, x):
            """
            Build combined vector. Fused arccos to halve trig cost:
              phi_dir and euc both need arccos(x)   — computed once as `a`
              phi_mag needs arccos(|x|)             — one separate call
            Saves one arccos per element vs naive implementation.
            """
            eps  = self.eps
            wb   = self._branch_weights()
            wf   = self._feat_weights()
            sqwf = wf.sqrt()

            # Fused: arccos(x) computed once, reused for phi_dir and euc
            a     = torch.acos(torch.clamp(x, -1+eps, 1-eps))
            phi_d = a * sqwf
            euc   = torch.tanh(a * 0.8) * sqwf

            # phi_mag needs arccos(|x|) — separate but unavoidable
            phi_m = torch.acos(
                torch.clamp(x.abs(), 0.0, 1-eps)) * sqwf

            mu  = x.mean(1, keepdim=True)
            sd  = x.std(1, keepdim=True).clamp(min=eps)
            shp = (x - mu) / sd

            return torch.cat([
                wb[0].sqrt() * phi_d,
                wb[1].sqrt() * phi_m,
                wb[2].sqrt() * euc,
                wb[3].sqrt() * shp,
            ], dim=1)

        def _distance_matrix(self, x, y):
            """
            L2 distance via torch.cdist. When y is the centroid buffer,
            uses the pre-computed centroid cache (set once per epoch in
            the training loop) to avoid recomputing K centroid feature
            vectors on every mini-batch.
            """
            fx = self._combined_features(x)
            # Use cache if y is the centroid buffer and cache is warm
            if y is self.centroids and self._centroid_cache is not None:
                fy = self._centroid_cache
            else:
                fy = self._combined_features(y)
            return torch.cdist(fx, fy, p=2).clamp(min=self.eps)

        def forward(self, x):
            """
            Pure AVM with gravity applied.

            Per-class tau: tau is (K,) when per_class_tau=True.
            d is (N, K). Division d/tau broadcasts (N,K)/(K,) = (N,K)
            so each class gets its own scoring sharpness.
            """
            tau = torch.exp(self.log_tau)          # (K,) or scalar
            d   = self._distance_matrix(x, self.centroids)   # (N, K)
            raw = 1.0 / (d / tau + self.eps)       # broadcasts correctly
            proba = raw / raw.sum(-1, keepdim=True)
            proba = proba * self.grav_w.unsqueeze(0)
            return proba / proba.sum(-1, keepdim=True)

    # ── loss ──────────────────────────────────────────────────────────────────

    def _ordinal_loss(self, probs, targets):
        """
        Earth Mover's Distance (Wasserstein-1) for ordered classes.

        For K ordered classes, the EMD between a predicted distribution p
        and a one-hot target at class y is the area between their CDFs:

            EMD = sum_{k=0}^{K-2} |CDF_p[k] - CDF_y[k]|

        where CDF_y[k] = 1 if k >= y (step function), 0 otherwise.

        Properties:
          - Predicting class y+1 when truth is y costs 1 unit
          - Predicting class y+2 when truth is y costs 2 units
          - Correct prediction costs 0 (CDFs are identical)
          - Naturally handles soft probabilities — no argmax needed

        This is differentiable and directly penalizes ordinal distance,
        unlike NLL which treats all wrong predictions equally.
        """
        N, K   = probs.shape
        # CDF of predictions: cumulative sum along class axis, drop last
        cdf_p  = torch.cumsum(probs, dim=1)[:, :-1]              # (N, K-1)
        # CDF of one-hot target: step function at target class
        # CDF_y[k] = 1 if target > k, else 0  →  vectorised via broadcast
        k_idx  = torch.arange(K - 1, device=targets.device)      # (K-1,)
        cdf_y  = (targets.unsqueeze(1) > k_idx.unsqueeze(0)).float()  # (N, K-1)
        return torch.abs(cdf_p - cdf_y).mean()

    def _loss(self, probs, targets, class_weights):
        K   = probs.shape[1]
        nll = nn.NLLLoss(weight=class_weights)(
                  torch.log(probs.clamp(1e-10)), targets)
        s   = self.label_smoothing / K
        ce  = (1 - self.label_smoothing + s) * nll + s * (
                   -torch.log(probs.clamp(1e-10)).mean())
        fw_reg = self.feat_weight_reg * (
            self.net_._feat_weights() - 1.0).pow(2).mean()

        if self.ordinal and K > 2:
            # Blend NLL-based CE with ordinal EMD loss.
            # Only meaningful for K>2 ordered classes — binary problems
            # have no ordinal distance to exploit.
            emd = self._ordinal_loss(probs, targets)
            ce  = (1.0 - self.ordinal_weight) * ce + self.ordinal_weight * emd

        return ce + fw_reg

    def _val_score(self, y_true, y_pred):
        """
        Composite validation metric used for both early stopping and
        lambda search. Blends accuracy, macro F1, and weighted F1 with
        configurable weights.

        Why composite rather than macro F1 alone:
          Macro F1 can be gamed by over-correcting minority classes —
          gravity, ordinal loss, and class-weighted NLL all push in the
          same direction. A model that predicts the rare class too often
          gains macro F1 while losing accuracy and weighted F1. The
          composite creates tension: corrections that improve macro F1
          only survive if they don't hurt accuracy or weighted F1 too much.

        Preset weight profiles via suggest_kwargs():
          Balanced data    → (0.34, 0.33, 0.33) — all three equal
          Moderate imbal.  → (0.20, 0.50, 0.30) — macro F1 prioritised
          Severe imbal.    → (0.40, 0.30, 0.30) — accuracy anchors the score
        """
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average='macro',    zero_division=0)
        wf1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return (self.val_acc_weight    * acc +
                self.val_macro_weight  * mf1 +
                self.val_weighted_weight * wf1)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        self.classes_ = np.unique(y)
        K      = len(self.classes_)
        n_feat = X.shape[1]

        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([label_to_idx[c] for c in y], dtype=np.int64)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Train / val split — enforce stratified minimum to prevent
        # degenerate validation sets that break early stopping and lambda search
        min_val = max(K * 5, int(len(X) * 0.05))   # at least 5 per class
        requested = int(len(X) * self.val_fraction)
        n_val = max(min_val, requested) if self.val_fraction > 0 else min_val

        if self.val_fraction == 0 and self.verbose:
            print(f"  val_fraction=0: using minimum val set of {n_val} samples"
                  f" ({n_val/len(X)*100:.1f}%) for early stopping.")
        # Stratified val split — ensures rare classes appear in validation
        # Random unbalanced split causes val_f1=0 on imbalanced datasets
        val_indices, tr_indices = [], []
        for cls in range(K):
            cls_idx = np.where(y_enc == cls)[0]
            rng.shuffle(cls_idx)
            n_cls_val = max(2, int(n_val * len(cls_idx) / len(X)))
            val_indices.append(cls_idx[:n_cls_val])
            tr_indices.append(cls_idx[n_cls_val:])
        va_idx = np.concatenate(val_indices)
        tr_idx = np.concatenate(tr_indices)
        rng.shuffle(tr_idx)
        X_tr, y_tr = X[tr_idx], y_enc[tr_idx]
        X_va, y_va = X[va_idx], y_enc[va_idx]

        self._norm_fit(X_tr)
        X_tr_n = self._norm_apply(X_tr)
        X_va_n = self._norm_apply(X_va)

        def tt(a, long=False):
            return torch.tensor(a,
                dtype=torch.long if long else torch.float32)

        # Class-weighted NLL — halve cap when gravity active to prevent
        # double-correction (NLL upweights minority, gravity does too)
        effective_cap = (self.weight_cap / 2.0
                         if self.gravity > 0.0 else self.weight_cap)
        cw   = compute_class_weight('balanced', classes=self.classes_,
                                    y=y[tr_idx])
        cw_t = torch.tensor(cw, dtype=torch.float32).clamp(
            max=effective_cap)
        cw_t = cw_t / cw_t.sum() * K

        # Centroids
        centroids = tt(np.stack([
            X_tr_n[y_tr == i].mean(0) for i in range(K)]))

        # Gravity weights — computed before net so grav_w can be set
        n_tr    = len(y_tr)
        counts  = np.array([(y_tr == i).sum() for i in range(K)],
                            dtype=np.float32)
        balanced = n_tr / (K * counts)
        balanced = balanced / balanced.mean()
        balanced = np.clip(balanced,
                           1.0 / self.gravity_cap, self.gravity_cap)
        balanced = balanced / balanced.mean()
        grav_w   = balanced.astype(np.float32)     # (K,)

        # Build net and inject gravity weights
        self.net_ = self._Net(centroids, K, n_feat,
                              per_class_tau=self.per_class_tau)
        if self.gravity != 0.0:
            self.net_.grav_w = torch.tensor(
                grav_w ** self.gravity, dtype=torch.float32)

        opt = optim.Adam(self.net_.parameters(),
                         lr=self.lr, weight_decay=self.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                  opt, T_0=30, T_mult=2, eta_min=1e-5)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                tt(X_tr_n), tt(y_tr, long=True)),
            batch_size=self.batch_size, shuffle=True)

        X_va_t = tt(X_va_n)
        best_f1, best_state, wait = -1.0, None, 0
        WARMUP = 20

        for epoch in range(self.epochs):
            self.net_.train()
            # Cache centroid features once per epoch — weights change per
            # batch so we recompute after each step, but computing once
            # per batch (inside _distance_matrix) wastes K forward passes
            # when K << batch_size. Caching here saves K * n_batches - 1
            # redundant centroid feature computations per epoch.
            with torch.no_grad():
                self.net_._centroid_cache = self.net_._combined_features(
                    self.net_.centroids)

            for xb, yb in loader:
                opt.zero_grad()
                self._loss(self.net_(xb), yb, cw_t).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.net_.parameters(), 1.0)
                opt.step()
                # Invalidate cache after weight update
                self.net_._centroid_cache = None
            sch.step()

            # Validation — forward() already applies gravity, so F1 here
            # matches the inference pipeline exactly
            self.net_.eval()
            with torch.no_grad():
                vp = self.net_(X_va_t).numpy()
            vf1 = self._val_score(y_va, vp.argmax(1))

            if vf1 > best_f1:
                best_f1    = vf1
                best_state = {k: v.clone() for k, v
                              in self.net_.state_dict().items()}
                wait = 0
            else:
                if epoch >= WARMUP:
                    wait += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                bw = self.net_._branch_weights().detach().numpy()
                print(f"  epoch {epoch+1:3d}  val_f1={vf1:.4f}"
                      f"  branches=["
                      f"d:{bw[0]:.2f} m:{bw[1]:.2f}"
                      f" e:{bw[2]:.2f} s:{bw[3]:.2f}]")

            if wait >= self.patience:
                if self.verbose:
                    print(f"  early stopping at epoch {epoch+1}")
                break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.best_val_f1_ = best_f1

        # Extract learned parameters
        with torch.no_grad():
            bw = self.net_._branch_weights().numpy().copy()
            fw = self.net_._feat_weights().numpy().copy()
            self.branch_weights_ = bw
            self.feat_weights_   = fw
            tau_raw = torch.exp(self.net_.log_tau)
            self.tau_ = tau_raw.numpy().copy()   # (K,) or scalar ndarray

        # Store gravity weights
        self.gravity_weights_ = grav_w

        # Build numpy combined vector function
        # Copy arrays into closure to avoid stale-reference bugs
        _bw = bw.copy(); _fw = fw.copy()
        s_d = np.float32(np.sqrt(_bw[0])); s_m = np.float32(np.sqrt(_bw[1]))
        s_e = np.float32(np.sqrt(_bw[2])); s_s = np.float32(np.sqrt(_bw[3]))
        sqfw = np.sqrt(_fw).astype(np.float32)

        def _make_combined(xn):
            eps   = 1e-7
            phi_d = np.arccos(np.clip(xn, -1+eps, 1-eps)) * sqfw
            phi_m = np.arccos(np.clip(np.abs(xn), 0.0, 1-eps)) * sqfw
            euc   = np.tanh(np.arccos(
                np.clip(xn, -1+eps, 1-eps)) * 0.8) * sqfw
            mu    = xn.mean(1, keepdims=True)
            sd    = xn.std(1,  keepdims=True).clip(min=eps)
            shp   = (xn - mu) / sd
            return np.concatenate([
                s_d*phi_d, s_m*phi_m, s_e*euc, s_s*shp
            ], axis=1).astype(np.float32)

        self._make_combined = _make_combined

        # Centroids in combined space
        self.centroids_combined_ = _make_combined(
            self.net_.centroids.numpy())

        # FAISS index
        self.lambda_ = self.lam_init
        if not _HAS_FAISS:
            if self.verbose:
                print("  faiss not installed — KNN disabled, lambda=1.0")
            self.lambda_ = 1.0
            self.index_  = None
        else:
            X_tr_c = _make_combined(X_tr_n)
            dim    = X_tr_c.shape[1]
            if self.use_ivf and X_tr_c.shape[0] >= 78:
                nlist = max(1, min(self.nlist, X_tr_c.shape[0] // 39))
                quantizer = faiss.IndexFlatL2(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                           faiss.METRIC_L2)
                index.train(X_tr_c)
                index.nprobe = self.nprobe
            else:
                index = faiss.IndexFlatL2(dim)
            index.add(X_tr_c)
            self.index_           = index
            self.train_class_idx_ = y_tr.copy()

            # Post-training lambda grid search on validation set
            # Uses full AVM + KNN + gravity — same as inference
            if self.auto_lambda:
                X_va_c  = _make_combined(X_va_n)
                avm_val = self._avm_proba(X_va_c)
                knn_val = self._knn_proba(X_va_c)

                best_lam_f1, best_lam = -1.0, self.lam_init
                # Start from 0.3 — pure KNN (lam=0) discards AVM geometry
                step = self.lam_search_step
                for lam_c in np.arange(0.3, 1.0 + step/2, step):
                    lam_c = float(np.round(lam_c, 6))
                    p = lam_c * avm_val + (1.0 - lam_c) * knn_val
                    if self.gravity != 0.0:
                        p = p * (grav_w[None, :] ** self.gravity)
                        p = p / p.sum(1, keepdims=True)
                    f1 = self._val_score(y_va, p.argmax(1))
                    if f1 > best_lam_f1:
                        best_lam_f1 = f1
                        best_lam    = lam_c

                self.lambda_ = best_lam
                if self.verbose:
                    print(f"  lambda search: best={best_lam:.2f}"
                          f"  val_f1={best_lam_f1:.4f}")

        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def _avm_proba(self, combined):
        a_sq = (combined                ** 2).sum(1, keepdims=True)
        b_sq = (self.centroids_combined_** 2).sum(1)
        ab   = combined @ self.centroids_combined_.T
        d    = np.sqrt(np.maximum(
            a_sq + b_sq[None, :] - 2.0 * ab, 0.0)).astype(np.float32)
        raw  = 1.0 / (d + 1e-10)
        return raw / raw.sum(1, keepdims=True)

    def _knn_proba(self, combined):
        if self.index_ is None:
            raise RuntimeError("FAISS index not built.")
        dist, idx = self.index_.search(combined, self.k)
        w = 1.0 / (dist + 1e-10)
        w = w / w.sum(1, keepdims=True)
        n_test = combined.shape[0]
        K      = len(self.classes_)
        proba  = np.zeros((n_test, K), dtype=np.float32)
        t_idx  = np.repeat(np.arange(n_test), self.k)
        np.add.at(proba,
                  (t_idx, self.train_class_idx_[idx.ravel()]),
                  w.ravel())
        return proba

    def predict_proba(self, X):
        check_is_fitted(self, 'centroids_combined_')
        X        = np.asarray(X, dtype=np.float32)
        combined = self._make_combined(self._norm_apply(X))

        avm = self._avm_proba(combined)
        if self.lambda_ >= 1.0 or self.index_ is None:
            proba = avm
        else:
            proba = (self.lambda_ * avm
                     + (1.0 - self.lambda_) * self._knn_proba(combined))

        if self.gravity != 0.0:
            proba = proba * (self.gravity_weights_[None, :] ** self.gravity)
            proba = proba / proba.sum(1, keepdims=True)

        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def summary(self, feature_names=None):
        check_is_fitted(self, 'branch_weights_')
        names = (list(feature_names) if feature_names is not None
                 else [f"f{i}" for i in range(len(self.feat_weights_))])
        bw  = self.branch_weights_
        lam = getattr(self, 'lambda_', self.lam_init)

        # Tau: scalar or per-class array
        tau = self.tau_
        if np.ndim(tau) == 0 or np.size(tau) == 1:
            tau_str = f"{float(tau):.3f}"
        else:
            tau_str = ("[" + " ".join(
                f"cls{i}:{v:.3f}" for i, v in enumerate(tau)) + "]")

        lines = [
            f"  tau={tau_str}  lambda={lam:.3f}"
            f"  (auto={self.auto_lambda})"
            f"  gravity={self.gravity:.2f}",
            f"  ordinal={self.ordinal}"
            f"  per_class_tau={self.per_class_tau}",
            f"  Branch weights: dir={bw[0]:.3f} mag={bw[1]:.3f}"
            f" euc={bw[2]:.3f} shp={bw[3]:.3f}",
        ]
        order = np.argsort(self.feat_weights_)[::-1]
        lines.append("  Top feature weights:")
        for i in order[:5]:
            lines.append(f"    {names[i]:<26} {self.feat_weights_[i]:.4f}")
        if self.gravity != 0.0:
            gw  = self.gravity_weights_
            gws = "  ".join(f"class{c}={w:.2f}"
                            for c, w in zip(self.classes_, gw))
            lines.append(
                f"  Gravity weights (cap={self.gravity_cap}): {gws}")
        return "\n".join(lines)
