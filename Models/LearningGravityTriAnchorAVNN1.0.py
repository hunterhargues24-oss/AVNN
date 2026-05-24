"""
GravityTriAnchorAVNN
=====================

Learnable version of TriAnchorAVNN. Optimises branch weights, per-feature
weights, tau, and lambda via PyTorch while preserving the four-branch
[-1,1] geometry and gravity calibration.

Four branches (same as TriAnchorAVNN)
---------------------------------------
  phi_dir_f  = arccos(x_f)       direction-aware angle      [0, π]
  phi_mag_f  = arccos(|x_f|)     magnitude angle (symmetric) [0, π/2]
  euc_f      = tanh(arccos(x_f) * 0.8)   nonlinear transform
  shape(x)   = (x - mean(x)) / std(x)    per-sample z-score

Learned parameters
-------------------
  raw_w_d/m/e/s   softmax → branch weights (4)
  raw_feat_w      softmax*F → per-feature weights shared across
                  direction, magnitude, and euclidean branches
  log_tau         exp → AVM scoring temperature
  raw_lambda      sigmoid → AVM vs KNN blend at inference

Training strategy
------------------
  Pure AVM during training — no KNN in the forward pass.
  Each epoch is O(batch × K × F) where K = num classes.
  KNN is added back at inference via FAISS (lam < 1.0).
  This matches BranchAdaptiveAVNN's proven speed pattern.

Gravity
--------
  Applied at inference as a post-hoc calibration step, not learned.
  Class-weighted NLL already handles minority class balance during
  training. Gravity then applies a static inference-time correction.
  gravity=0.0 → neutral. gravity=0.5 → minority boost (recommended
  for imbalanced datasets).
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


class GravityTriAnchorAVNN(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    k : int, default=5
    lr : float, default=1e-3
    epochs : int, default=300
    batch_size : int, default=64
    patience : int, default=60
    val_fraction : float, default=0.15
    label_smoothing : float, default=0.05
    weight_cap : float, default=1.5
        Cap on class weights in NLL loss.
    weight_decay : float, default=1e-4
    gravity : float, default=0.0
        Inference-time minority class boost. 0=neutral, 0.5=moderate boost.
    gravity_cap : float, default=1.5
        Symmetric cap on gravity weights — prevents majority class
        from being suppressed below 1/gravity_cap.
    lam : float, default=0.7
        AVM vs KNN blend at inference. 1.0 = pure AVM, no FAISS needed.
    use_ivf : bool, default=True
    nlist : int, default=1000
    nprobe : int, default=10
    random_state : int or None, default=None
    verbose : bool, default=False
    """

    def __init__(self, k=5, lr=1e-3, epochs=300, batch_size=64,
                 patience=60, val_fraction=0.15, label_smoothing=0.05,
                 weight_cap=1.5, weight_decay=1e-4,
                 gravity=0.0, gravity_cap=1.5, lam=0.7,
                 use_ivf=True, nlist=1000, nprobe=10,
                 random_state=None, verbose=False):
        self.k               = k
        self.lr              = lr
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.patience        = patience
        self.val_fraction    = val_fraction
        self.label_smoothing = label_smoothing
        self.weight_cap      = weight_cap
        self.weight_decay    = weight_decay
        self.gravity         = gravity
        self.gravity_cap     = gravity_cap
        self.lam             = lam
        self.use_ivf         = use_ivf
        self.nlist           = nlist
        self.nprobe          = nprobe
        self.random_state    = random_state
        self.verbose         = verbose

    # ── preprocessing ─────────────────────────────────────────────────────────

    def _norm_fit(self, X):
        self.mn_  = X.min(0)
        self.rng_ = np.maximum(X.max(0) - self.mn_, 1e-10)

    def _norm_apply(self, X):
        return (-1.0 + 2.0 * (X - self.mn_) / self.rng_).astype(np.float32)

    # ── inner PyTorch model ───────────────────────────────────────────────────

    class _Net(nn.Module):

        def __init__(self, c_norm, K, n_feat, eps=1e-7):
            super().__init__()
            self.K, self.n_feat, self.eps = K, n_feat, eps

            # Centroids as a buffer (fixed — class means don't change)
            self.register_buffer("c_norm", c_norm)

            # Learned parameters
            self.log_tau     = nn.Parameter(torch.tensor(-0.5))
            self.raw_lambda  = nn.Parameter(torch.tensor(0.85))  # inference only
            self.raw_w_d     = nn.Parameter(torch.tensor(0.0))
            self.raw_w_m     = nn.Parameter(torch.tensor(0.0))
            self.raw_w_e     = nn.Parameter(torch.tensor(0.0))
            self.raw_w_s     = nn.Parameter(torch.tensor(0.0))
            self.raw_feat_w  = nn.Parameter(torch.zeros(n_feat))

        # ── parameter accessors ───────────────────────────────────────────

        def _branch_w(self):
            return torch.softmax(torch.stack([
                self.raw_w_d, self.raw_w_m,
                self.raw_w_e, self.raw_w_s]), dim=0)

        def _feat_w(self):
            w = torch.softmax(self.raw_feat_w, dim=0)
            return w * self.n_feat   # scaled so uniform = 1.0 each

        # ── branch computations ───────────────────────────────────────────

        @staticmethod
        def _phi_dir(x, eps):
            return torch.acos(torch.clamp(x, -1.0 + eps, 1.0 - eps))

        @staticmethod
        def _phi_mag(x, eps):
            return torch.acos(torch.clamp(x.abs(), 0.0, 1.0 - eps))

        @staticmethod
        def _euc(x, eps):
            return torch.tanh(
                torch.acos(torch.clamp(x, -1.0 + eps, 1.0 - eps)) * 0.8)

        @staticmethod
        def _shape(x, eps):
            mu = x.mean(1, keepdim=True)
            sd = x.std(1, keepdim=True).clamp(min=eps)
            return (x - mu) / sd

        # ── distance: (N, F) queries vs (M, F) references → (N, M) ───────

        def _dist(self, xn_q, xn_r):
            """
            Compute the SAME distance as the inference combined vector.

            Combined vector distance:
                ||combined_q - combined_r||^2
                  = w_d * ||sqfw * phi_dir_q - sqfw * phi_dir_r||^2
                  + w_m * ||sqfw * phi_mag_q - sqfw * phi_mag_r||^2
                  + w_e * ||sqfw * euc_q     - sqfw * euc_r    ||^2
                  + w_s * ||shape_q          - shape_r         ||^2

            Using L2 (not L1) per branch so training and inference are
            identical. Previous L1 formulation caused branch weights to
            diverge wildly (shape→0, direction→0.43) because the model
            was learning weights for one metric but being tested on another.
            """
            eps  = self.eps
            wb   = self._branch_w()       # (4,)
            wf   = self._feat_w()         # (F,)
            sqwf = wf.sqrt()

            # Scale features by sqrt(feat_weight) — same as combined vector
            pd_q = self._phi_dir(xn_q, eps) * sqwf
            pd_r = self._phi_dir(xn_r, eps) * sqwf
            pm_q = self._phi_mag(xn_q, eps) * sqwf
            pm_r = self._phi_mag(xn_r, eps) * sqwf
            eu_q = self._euc(xn_q, eps)    * sqwf
            eu_r = self._euc(xn_r, eps)    * sqwf
            sh_q = self._shape(xn_q, eps)
            sh_r = self._shape(xn_r, eps)

            # Squared L2 per branch
            sq_d = ((pd_q.unsqueeze(1) - pd_r.unsqueeze(0))**2).sum(-1)
            sq_m = ((pm_q.unsqueeze(1) - pm_r.unsqueeze(0))**2).sum(-1)
            sq_e = ((eu_q.unsqueeze(1) - eu_r.unsqueeze(0))**2).sum(-1)
            sq_s = ((sh_q.unsqueeze(1) - sh_r.unsqueeze(0))**2).sum(-1)

            # Branch-weighted sum then sqrt — matches combined vector L2
            return torch.sqrt(
                wb[0]*sq_d + wb[1]*sq_m + wb[2]*sq_e + wb[3]*sq_s + eps)

        def forward(self, xn):
            """Pure AVM — training only. Fast: O(N × K × F)."""
            tau = torch.exp(self.log_tau)
            d   = self._dist(xn, self.c_norm)
            raw = 1.0 / (d / tau + self.eps)
            return raw / raw.sum(-1, keepdim=True)

    # ── loss ──────────────────────────────────────────────────────────────────

    def _loss(self, probs, targets, class_w):
        K   = probs.shape[1]
        nll = nn.NLLLoss(weight=class_w)(
                  torch.log(probs.clamp(1e-10)), targets)
        s   = self.label_smoothing / K
        return (1 - self.label_smoothing + s) * nll + s * (
                   -torch.log(probs.clamp(1e-10)).mean())

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        rng           = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        K             = len(self.classes_)
        n_feat        = X.shape[1]

        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_enc        = np.array([label_to_idx[c] for c in y], dtype=np.int64)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Train / val split
        n_val  = max(K, int(len(X) * self.val_fraction))
        perm   = rng.permutation(len(X))
        va_idx, tr_idx = perm[:n_val], perm[n_val:]
        X_tr, y_tr = X[tr_idx], y_enc[tr_idx]
        X_va, y_va = X[va_idx], y_enc[va_idx]

        self._norm_fit(X_tr)
        Xtrn = self._norm_apply(X_tr)
        Xvan = self._norm_apply(X_va)

        def tt(a, long=False):
            return torch.tensor(a,
                dtype=torch.long if long else torch.float32)

        # Class-weighted NLL
        # Fix 2: when gravity > 0, class-weighted NLL and gravity both
        # upweight minority classes. Halve the effective weight cap so
        # NLL handles half the correction and gravity handles the other half.
        effective_cap = (self.weight_cap / 2.0
                         if self.gravity > 0.0 else self.weight_cap)
        cw   = compute_class_weight('balanced', classes=self.classes_,
                                    y=y[tr_idx])
        cw_t = torch.tensor(cw, dtype=torch.float32).clamp(
            max=effective_cap)
        cw_t = cw_t / cw_t.sum() * K

        # Centroids from class means
        c_norm = tt(np.stack([
            Xtrn[y_tr == i].mean(0) for i in range(K)]))

        self.net_ = self._Net(c_norm, K, n_feat)

        # Compute gravity weights now so validation can use them
        # Fix 1: early stopping scores the full inference pipeline
        # (AVM + gravity), not just raw AVM probabilities.
        n_tr    = len(y_tr)
        counts  = np.array([(y_tr == i).sum() for i in range(K)],
                            dtype=np.float32)
        balanced = n_tr / (K * counts)
        balanced = balanced / balanced.mean()
        balanced = np.clip(balanced,
                           1.0 / self.gravity_cap, self.gravity_cap)
        balanced = balanced / balanced.mean()
        grav_w   = balanced.astype(np.float32)          # (K,) for validation

        opt = optim.Adam(self.net_.parameters(),
                         lr=self.lr, weight_decay=self.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                  opt, T_0=30, T_mult=2, eta_min=1e-5)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                tt(Xtrn), tt(y_tr, long=True)),
            batch_size=self.batch_size, shuffle=True)

        Xvan_t = tt(Xvan)
        best_f1, best_state, wait = -1.0, None, 0
        WARMUP = 20

        for epoch in range(self.epochs):
            self.net_.train()
            for xn, yb in loader:
                opt.zero_grad()
                self._loss(self.net_(xn), yb, cw_t).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.net_.parameters(), 1.0)
                opt.step()
            sch.step()

            self.net_.eval()
            with torch.no_grad():
                val_proba = self.net_(Xvan_t).numpy()
            # Fix 1: apply gravity to validation probabilities so early
            # stopping optimises the same pipeline used at inference.
            if self.gravity != 0.0:
                val_proba = val_proba * (grav_w[None, :] ** self.gravity)
                val_proba = val_proba / val_proba.sum(1, keepdims=True)
            pred = val_proba.argmax(1)
            vf1  = f1_score(y_va, pred, average='macro', zero_division=0)

            if vf1 > best_f1:
                best_f1    = vf1
                best_state = {k: v.clone() for k, v
                              in self.net_.state_dict().items()}
                wait = 0
            else:
                if epoch >= WARMUP:
                    wait += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                bw = self.net_._branch_w().detach().numpy()
                print(f"  epoch {epoch+1:3d}  val_f1={vf1:.4f}"
                      f"  branches=["
                      f"dir:{bw[0]:.2f} mag:{bw[1]:.2f}"
                      f" euc:{bw[2]:.2f} shp:{bw[3]:.2f}]")

            if wait >= self.patience:
                if self.verbose:
                    print(f"  early stopping at epoch {epoch+1}")
                break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.best_val_f1_ = best_f1

        # Extract learned parameters
        with torch.no_grad():
            bw = self.net_._branch_w().numpy()
            self.branch_weights_  = {
                'direction': bw[0], 'magnitude': bw[1],
                'euclidean': bw[2], 'shape':     bw[3]}
            self.feat_weights_    = self.net_._feat_w().numpy()
            self.tau_             = torch.exp(self.net_.log_tau).item()
            self.lambda_          = torch.sigmoid(
                self.net_.raw_lambda).item()
            self.centroids_norm_  = self.net_.c_norm.numpy().copy()

        # ── Build FAISS index for KNN inference ───────────────────────
        # Combined vector uses learned weights — same construction as
        # static TriAnchorAVNN but with learned sqrt scales.
        s_d = np.float32(np.sqrt(bw[0]))
        s_m = np.float32(np.sqrt(bw[1]))
        s_e = np.float32(np.sqrt(bw[2]))
        s_s = np.float32(np.sqrt(bw[3]))
        fw  = self.feat_weights_.astype(np.float32)
        sqfw = np.sqrt(fw)

        def _make_combined(xn):
            eps   = 1e-7
            phi_d = np.arccos(np.clip(xn, -1+eps, 1-eps)) * sqfw
            phi_m = np.arccos(np.clip(np.abs(xn), 0, 1-eps)) * sqfw
            euc   = np.tanh(np.arccos(
                np.clip(xn, -1+eps, 1-eps)) * 0.8) * sqfw
            mu    = xn.mean(1, keepdims=True)
            sd    = xn.std(1, keepdims=True).clip(min=eps)
            shp   = (xn - mu) / sd
            return np.concatenate([
                s_d*phi_d, s_m*phi_m, s_e*euc, s_s*shp
            ], axis=1).astype(np.float32)

        self._make_combined = _make_combined

        # Centroids in combined space
        self.centroids_combined_ = _make_combined(
            self.centroids_norm_).astype(np.float32)

        # Gravity weights already computed above (used during validation)
        self.gravity_weights_ = grav_w

        # FAISS
        if self.lam < 1.0:
            if not _HAS_FAISS:
                raise ImportError(
                    "faiss not installed. Set lam=1.0 for pure AVM.")
            combined = _make_combined(Xtrn)
            dim      = combined.shape[1]
            if self.use_ivf and combined.shape[0] >= 78:
                nlist = max(1, min(self.nlist,
                                   combined.shape[0] // 39))
                quantizer = faiss.IndexFlatL2(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                            faiss.METRIC_L2)
                index.train(combined)
                index.nprobe = self.nprobe
            else:
                index = faiss.IndexFlatL2(dim)
            index.add(combined)
            self.index_           = index
            self.train_class_idx_ = y_tr.copy()

        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _avm_proba(self, combined):
        a_sq = (combined                ** 2).sum(1, keepdims=True)
        b_sq = (self.centroids_combined_** 2).sum(1)
        ab   = combined @ self.centroids_combined_.T
        d    = np.sqrt(np.maximum(
            a_sq + b_sq[None,:] - 2.0*ab, 0.0)).astype(np.float32)
        raw  = 1.0 / (d + 1e-10)
        return raw / raw.sum(1, keepdims=True)

    def _knn_proba(self, combined):
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
        Xn       = self._norm_apply(X)
        combined = self._make_combined(Xn)

        avm   = self._avm_proba(combined)
        proba = avm if self.lam >= 1.0 else (
            self.lam * avm + (1.0 - self.lam) * self._knn_proba(combined))

        if self.gravity != 0.0:
            proba = proba * (self.gravity_weights_[None,:] ** self.gravity)
            proba = proba / proba.sum(1, keepdims=True)

        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def summary(self, feature_names=None):
        check_is_fitted(self, 'branch_weights_')
        names = (list(feature_names) if feature_names
                 else [f"f{i}" for i in range(len(self.feat_weights_))])
        bw = self.branch_weights_
        lines = [
            f"  tau={self.tau_:.3f}  lambda={self.lambda_:.3f}"
            f"  gravity={self.gravity:.2f}",
            f"  Branch weights:"
            f" direction={bw['direction']:.3f}"
            f" magnitude={bw['magnitude']:.3f}"
            f" euclidean={bw['euclidean']:.3f}"
            f" shape={bw['shape']:.3f}",
        ]
        fw    = self.feat_weights_
        order = np.argsort(fw)[::-1]
        lines.append(f"  Top feature weights:")
        for i in order[:5]:
            lines.append(f"    {names[i]:<26} {fw[i]:.4f}")
        if self.gravity != 0.0:
            gw  = self.gravity_weights_
            gws = "  ".join(f"class{c}={w:.2f}"
                            for c, w in zip(self.classes_, gw))
            lines.append(
                f"  Gravity weights (cap={self.gravity_cap}): {gws}")
        return "\n".join(lines)
