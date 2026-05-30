"""
LearningAVNN
============
Geometric tabular classifier combining a trained Angular Vector Machine
(AVM) with a FAISS-accelerated KNN voter, blended at inference.

Architecture
------------
  Input → MinMax normalise [-1,1]
       → AVM: multi-branch combined vector (configurable via branch_set)
            → Mahalanobis inverse-distance to K×m learnable prototypes
              → AVM probability  (trained end-to-end)
       → KNN: linear + shape + quadratic combined vector
            → FAISS inverse-distance vote over k nearest training points
              → KNN probability  (fixed features, post-fit)
       → predict: λ·AVM + (1-λ)·KNN   (λ tuned on val, or per-sample
                                       confidence gate)

This is late-fusion. The AVM trains end-to-end. The KNN voter is built
from the same training data using a fixed geometric view once training
completes, then blended at inference.

AVM Branch Sets  (learned — global structure / centroid proximity)
------------------------------------------------------------------
  default      tac + bnd + cir         (3 branches, original)
  extended     tac + bnd + cir + sq + mtac   (5 branches)
  tac_family   tac + mtac + sq         (3 branches, all tac-shaped /
                                        symmetric, no boundary coupling)

  Branch shapes:
    tac     tanh(arccos(x)·0.8)         monotone decreasing, asymmetric,
                                        middle-sensitive, saturated extremes
    bnd     √(‖x‖²-2x+1)                cross-feature coupling via ‖x‖²;
                                        optional dual extends to 2F
    cir     √(1-x²)                     symmetric, peaks at x=0
    sq      x²                          symmetric, peaks at x=±1 (opposite
                                        curvature from cir)
    mtac    tanh(arccos(-x)·0.8)        mirror of tac — monotone INCREASING,
                                        captures positive-direction signal

Performance
-----------
  device='auto'      Picks 'cuda' if available, else 'cpu'.
  Mixed precision    Automatic on CUDA via torch.amp.autocast — roughly 2×
                     forward/backward throughput on modern GPUs.
  Index-on-demand    Shuffle indices once per epoch, slice X_tr_t inside
                     the batch loop. No full-tensor copy per epoch.
  Cached class means EMA path with m=1 reuses class means computed once
                     before training (they don't change).

Imbalance Handling
------------------
  SupCon loss     — pulls same-class AVM embeddings together per batch
  Per-class tau   — each class learns its own scoring sharpness
  Mahalanobis RDA — full per-class covariance at inference (α-blended QDA/LDA)
  Class weights   — balanced NLL capped at weight_cap
  Ordinal EMD     — Earth Mover's Distance for ordered class structures
  EMA centroids   — exponential moving average stabilises minority prototypes
  val_macro_bias  — early stopping biased toward macro F1 over accuracy

Key Parameters
--------------
  k              int    KNN neighbours
  n_prototypes   int    prototypes per class
  branch_set     str    'default', 'extended', or 'tac_family'
  device         str    'auto', 'cuda', 'cpu', or a specific device string
  mahal_alpha    float  RDA blend (0=QDA, 1=LDA)
  lam_floor      float  minimum AVM weight in lambda grid search
  val_macro_bias float  0=accuracy, 1=macro F1 for early stopping
  supcon         bool   supervised contrastive loss
  ema_centroids  bool   EMA prototype stabilisation
  ortho_reg      float  Gram matrix branch decorrelation
  use_dual_boundary bool  extend boundary branch to 2F
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


# ── Branch registry ──────────────────────────────────────────────────────────
# Single source of truth for branch definitions. Adding a new branch means
# adding to BRANCH_NAMES + the dispatch in _avm_features + the dispatch in
# _make_avm_combined (post-fit). Three places, no more.
BRANCH_SETS = {
    'default':    ['tac', 'b',    'cir'],
    'extended':   ['tac', 'b',    'cir', 'sq', 'mtac'],
    'tac_family': ['tac', 'mtac', 'sq'],
}

# Init values for raw gate parameters (pre-sigmoid).
BRANCH_INITS = {
    'tac':  0.5,   # +0.5 head start (original design)
    'b':    0.5,   # +0.5 head start (original design)
    'cir':  0.0,
    'sq':   0.0,   # let gradient decide
    'mtac': 0.0,   # let gradient decide
}

# Display names for summary()
BRANCH_DISPLAY = {
    'tac':  'tanh_arccos',
    'b':    'boundary',
    'cir':  'circular',
    'sq':   'squared',
    'mtac': 'mirror_tac',
}


def _resolve_device(spec):
    """Map 'auto' to cuda/cpu; otherwise pass through to torch.device."""
    if spec == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(spec)


class LearningAVNN(BaseEstimator, ClassifierMixin):

    def __init__(self, k=5, lr=4e-3, epochs=300, batch_size=1024,
                 patience=60, val_fraction=0.15, label_smoothing=0.05,
                 weight_cap=1.5, weight_decay=1e-4, feat_weight_reg=0.05,
                 centroid_reg=0.1, centroid_sep=0.05,
                 n_prototypes=1, intra_sep=0.05,
                 branch_set='default',
                 use_dual_boundary=False,
                 ortho_reg=0.01,
                 mahalanobis=True, mahal_reg=1e-6, mahal_alpha=0.5,
                 lam_floor=0.5, entropy_lambda=False,
                 lam_confident=0.90, lam_uncertain=0.40,
                 ordinal=False, ordinal_weight=0.5,
                 per_class_tau=True,
                 val_macro_bias=0.5,
                 use_ivf=True,
                 supcon=False, supcon_weight=0.3, supcon_temp=0.1,
                 ema_centroids=False, ema_beta=0.9,
                 device='auto', use_amp=True,
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
        self.centroid_reg        = centroid_reg
        self.centroid_sep        = centroid_sep
        self.n_prototypes        = n_prototypes
        self.intra_sep           = intra_sep
        self.branch_set          = branch_set
        self.use_dual_boundary   = use_dual_boundary
        self.ortho_reg           = ortho_reg
        self.mahalanobis         = mahalanobis
        self.mahal_reg           = mahal_reg
        self.mahal_alpha         = mahal_alpha
        self.lam_floor           = lam_floor
        self.entropy_lambda      = entropy_lambda
        self.lam_confident       = lam_confident
        self.lam_uncertain       = lam_uncertain
        self.ordinal             = ordinal
        self.ordinal_weight      = ordinal_weight
        self.per_class_tau       = per_class_tau
        self.val_macro_bias      = val_macro_bias
        self.use_ivf             = use_ivf
        self.supcon              = supcon
        self.supcon_weight       = supcon_weight
        self.supcon_temp         = supcon_temp
        self.ema_centroids       = ema_centroids
        self.ema_beta            = ema_beta
        self.device              = device
        self.use_amp             = use_amp
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

        def __init__(self, centroids, K, n_feat, m=1,
                     branch_set='default',
                     per_class_tau=False,
                     use_dual_boundary=False, eps=1e-7):
            super().__init__()
            self.K, self.m, self.n_feat, self.eps = K, m, n_feat, eps
            self.per_class_tau     = per_class_tau
            self.use_dual_boundary = use_dual_boundary
            self.branch_set        = branch_set
            self._sep_cache        = None

            if branch_set not in BRANCH_SETS:
                raise ValueError(
                    f"branch_set must be one of {list(BRANCH_SETS)}, "
                    f"got {branch_set!r}")
            self.active_branches = BRANCH_SETS[branch_set]

            self.centroids = nn.Parameter(centroids.clone())
            self.register_buffer("centroid_anchor", centroids.clone())

            if per_class_tau:
                self.log_tau = nn.Parameter(torch.full((K,), -0.5))
            else:
                self.log_tau = nn.Parameter(torch.tensor(-0.5))

            # AVM branch gates — one raw weight per active branch.
            # ParameterDict so we only create what we use (no dead weights
            # fighting weight_decay).
            self.raw_w = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(BRANCH_INITS[name]))
                for name in self.active_branches
            })

            # Feature weights — shared with the KNN voter post-fit
            self.raw_feat_w = nn.Parameter(torch.zeros(n_feat))

        def _avm_branch_weights(self):
            """Active branches — normalised sigmoid over active gates only."""
            raw   = torch.stack(
                [self.raw_w[name] for name in self.active_branches])
            gates = torch.sigmoid(raw)
            return gates / (gates.sum() + 1e-10)

        def _feat_weights(self):
            w = torch.softmax(self.raw_feat_w, dim=0)
            return w * self.n_feat   # uniform = 1.0 each

        @staticmethod
        def _boundary_dists(x, eps, dual=False):
            """
            Distances to hypercube face centres.
            dual=False: (N, F)  distances to +e_f only.
            dual=True:  (N, 2F) distances to +e_f and -e_f interleaved.
              pos: ||x - e_f||² = ||x||² - 2x_f + 1
              neg: ||x + e_f||² = ||x||² + 2x_f + 1
            """
            norm_sq     = (x**2).sum(dim=1, keepdim=True)
            dist_sq_pos = (norm_sq - 2.0*x + 1.0).clamp(min=eps)
            if not dual:
                return torch.sqrt(dist_sq_pos)
            dist_sq_neg = (norm_sq + 2.0*x + 1.0).clamp(min=eps)
            stacked = torch.stack([dist_sq_pos, dist_sq_neg], dim=2)
            return torch.sqrt(stacked.reshape(stacked.shape[0], -1))

        def _branch_feature(self, name, x, wf, sqwf):
            """
            Compute a single branch's per-feature output (pre-gate-weighting).
            One dispatch table keeps the torch path aligned with the post-fit
            numpy path (_np_branch_feature in fit()).
            """
            eps = self.eps
            if name == 'tac':
                a = torch.acos(torch.clamp(x, -1+eps, 1-eps))
                return torch.tanh(a * 0.8) * sqwf
            if name == 'b':
                return self._boundary_dists(x, eps, dual=self.use_dual_boundary)
            if name == 'cir':
                return torch.sqrt(torch.clamp(1.0 - x*x, min=eps)) * sqwf
            if name == 'sq':
                return (x * x) * sqwf
            if name == 'mtac':
                a = torch.acos(torch.clamp(-x, -1+eps, 1-eps))
                return torch.tanh(a * 0.8) * sqwf
            raise ValueError(f"unknown branch: {name}")

        def _avm_features(self, x, wf):
            """AVM combined vector — concat of all active branches, gated."""
            sqwf = wf.sqrt()
            wa   = self._avm_branch_weights()
            parts = []
            for i, name in enumerate(self.active_branches):
                feat = self._branch_feature(name, x, wf, sqwf)
                parts.append(wa[i].sqrt() * feat)
            return torch.cat(parts, dim=1)

        def forward(self, x):
            """
            AVM forward pass — scores distances in the active branch space
            to K*m learnable prototypes. Returns (proba, fw, embeddings).
            """
            wf   = self._feat_weights()
            N    = x.shape[0]
            K, m = self.K, self.m
            tau  = torch.exp(self.log_tau)

            fx_avm     = self._avm_features(x, wf)
            proto_flat = self.centroids.reshape(K*m, self.n_feat)
            fy_avm     = self._avm_features(proto_flat, wf)

            sq_x   = (fx_avm**2).sum(1, keepdim=True)
            sq_y   = (fy_avm**2).sum(1)
            dot    = fx_avm @ fy_avm.T
            d_flat = torch.sqrt(torch.clamp(
                sq_x + sq_y[None,:] - 2.0*dot, min=self.eps))
            d      = d_flat.reshape(N, K, m)

            tau_e  = tau.reshape(1,K,1) if (K>1 and tau.dim()>0) else tau
            raw    = (1.0 / (d / tau_e + self.eps)).sum(2)
            proba  = raw / raw.sum(-1, keepdim=True)

            return proba, wf, fx_avm

    # ── loss ──────────────────────────────────────────────────────────────────

    def _ordinal_loss(self, probs, targets):
        """Earth Mover's Distance — penalises ordinal mistakes proportionally."""
        N, K   = probs.shape
        cdf_p  = torch.cumsum(probs, dim=1)[:, :-1]
        k_idx  = torch.arange(K - 1, device=targets.device)
        cdf_y  = (targets.unsqueeze(1) > k_idx.unsqueeze(0)).float()
        return torch.abs(cdf_p - cdf_y).mean()

    def _supcon_loss(self, embeddings, targets):
        """
        Supervised Contrastive Loss (Khosla et al., 2020).
        Pulls same-class AVM embeddings together, pushes apart different.
        """
        z = nn.functional.normalize(embeddings, dim=1)
        N = z.shape[0]

        sim = (z @ z.T) / self.supcon_temp
        labels   = targets.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)

        n_pos = pos_mask.sum(1)
        valid = n_pos > 0

        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device)

        sim_max, _ = sim.max(dim=1, keepdim=True)
        exp_sim    = torch.exp(sim - sim_max.detach())

        eye        = torch.eye(N, device=embeddings.device)
        exp_sim    = exp_sim * (1.0 - eye)

        log_prob   = (sim - sim_max.detach()) - torch.log(
            exp_sim.sum(1, keepdim=True).clamp(min=1e-10))

        mean_log_prob = (pos_mask * log_prob).sum(1) / n_pos.clamp(min=1)
        return -mean_log_prob[valid].mean()

    def _loss(self, probs, targets, class_weights, fw=None, embeddings=None):
        K   = probs.shape[1]
        nll = nn.NLLLoss(weight=class_weights)(
                  torch.log(probs.clamp(1e-10)), targets)
        # Label smoothing: (1-α)*NLL + α * mean_k(-log p_k)
        smooth_term = -torch.log(probs.clamp(1e-10)).mean(dim=1).mean()
        ce  = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth_term

        _fw = fw if fw is not None else self.net_._feat_weights()
        fw_reg = self.feat_weight_reg * (_fw - 1.0).pow(2).mean()

        c_anchor = self.centroid_reg * (
            self.net_.centroids - self.net_.centroid_anchor
        ).pow(2).mean()

        # Centroid separation — cached per epoch, reused across batches.
        if self.net_._sep_cache is None:
            if self.net_.K > 1:
                _K, _m = self.net_.K, self.net_.m
                proto_flat = self.net_.centroids.reshape(_K * _m, self.net_.n_feat)
                c_dists    = torch.cdist(proto_flat, proto_flat, p=2)

                class_idx   = torch.arange(_K, device=c_dists.device).repeat_interleave(_m)
                diff_class  = (class_idx.unsqueeze(0) != class_idx.unsqueeze(1)).float()
                upper       = torch.triu(torch.ones_like(diff_class), diagonal=1)
                inter_mask  = diff_class * upper

                if inter_mask.sum() > 0:
                    inter = -self.centroid_sep * (
                        c_dists * inter_mask).sum() / inter_mask.sum()
                else:
                    inter = torch.tensor(0.0, device=c_dists.device)

                intra = torch.tensor(0.0, device=c_dists.device)
                if _m > 1 and self.intra_sep > 0.0:
                    for k in range(_K):
                        proto_k = self.net_.centroids[k]
                        d_intra = torch.cdist(proto_k, proto_k, p=2)
                        mask_k  = torch.triu(
                            torch.ones(_m, _m, device=d_intra.device), diagonal=1)
                        if mask_k.sum() > 0:
                            intra = intra - self.intra_sep * (
                                d_intra * mask_k).sum() / mask_k.sum()

                self.net_._sep_cache = (inter + intra).detach()
            else:
                self.net_._sep_cache = torch.tensor(
                    0.0, device=self.net_.centroids.device)
        c_sep = self.net_._sep_cache

        if self.ordinal and K >= 2:
            emd = self._ordinal_loss(probs, targets)
            ce  = (1.0 - self.ordinal_weight) * ce + self.ordinal_weight * emd

        if self.supcon and embeddings is not None:
            sc  = self._supcon_loss(embeddings, targets)
            ce  = (1.0 - self.supcon_weight) * ce + self.supcon_weight * sc

        if self.ortho_reg > 0.0 and embeddings is not None:
            n_branches = embeddings.shape[1] // self.net_.n_feat
            F_branch   = self.net_.n_feat
            chunks  = embeddings[:, :n_branches * F_branch].reshape(
                -1, n_branches, F_branch)
            normed  = nn.functional.normalize(chunks, dim=2)
            G       = torch.bmm(normed, normed.transpose(1, 2)).mean(0)
            I       = torch.eye(n_branches, device=G.device)
            ortho   = self.ortho_reg * (G - I).pow(2).sum()
            return ce + fw_reg + c_anchor + c_sep + ortho

        return ce + fw_reg + c_anchor + c_sep

    # ── composite validation metric ───────────────────────────────────────────

    def _val_score(self, y_true, y_pred):
        """
        Composite validation metric from a single confusion matrix.
        Faster than three sklearn metric calls.
        """
        from sklearn.metrics import confusion_matrix
        labels = np.arange(len(self.classes_))
        cm     = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
        tp     = np.diag(cm)
        total  = cm.sum()
        sup    = cm.sum(1)
        pred_s = cm.sum(0)

        acc  = tp.sum() / (total + 1e-10)
        prec = np.zeros_like(f1 := np.zeros(len(labels)))
        rec  = np.zeros_like(prec)
        np.divide(tp, pred_s, out=prec, where=pred_s > 0)
        np.divide(tp, sup,    out=rec,  where=sup    > 0)
        denom = prec + rec
        np.divide(2 * prec * rec, denom, out=f1, where=denom > 0)

        macro    = f1.mean()
        weighted = (f1 * sup).sum() / (total + 1e-10)

        b = self.val_macro_bias
        return (1 - b) * 0.5 * (acc + weighted) + b * macro

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        if self.branch_set not in BRANCH_SETS:
            raise ValueError(
                f"branch_set must be one of {list(BRANCH_SETS)}, "
                f"got {self.branch_set!r}")

        # ── Device resolution ────────────────────────────────────────────────
        device = _resolve_device(self.device)
        self.device_ = device
        use_amp = bool(self.use_amp) and device.type == 'cuda'

        self.classes_ = np.unique(y)
        K      = len(self.classes_)
        n_feat = X.shape[1]

        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([label_to_idx[c] for c in y], dtype=np.int64)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(self.random_state)

        # Stratified train/val split
        min_val  = max(K * 5, int(len(X) * 0.05))
        n_val    = max(min_val, int(len(X) * self.val_fraction))
        if self.verbose and self.val_fraction == 0:
            print(f"  val_fraction=0: using {n_val} val samples "
                  f"({n_val/len(X)*100:.1f}%)")

        val_idx, tr_idx = [], []
        for cls in range(K):
            idx = np.where(y_enc == cls)[0]
            rng.shuffle(idx)
            n_v = max(2, int(n_val * len(idx) / len(X)))
            val_idx.append(idx[:n_v])
            tr_idx.append(idx[n_v:])
        va_idx = np.concatenate(val_idx)
        tr_idx = np.concatenate(tr_idx)
        rng.shuffle(tr_idx)

        X_tr, y_tr = X[tr_idx], y_enc[tr_idx]
        X_va, y_va = X[va_idx], y_enc[va_idx]

        self._norm_fit(X_tr)
        X_tr_n = self._norm_apply(X_tr)
        X_va_n = self._norm_apply(X_va)

        # All training/val tensors created directly on device.
        def tt(a, long=False):
            return torch.tensor(a,
                dtype=torch.long if long else torch.float32,
                device=device)

        cw   = compute_class_weight('balanced', classes=self.classes_,
                                    y=y[tr_idx])
        cw_t = torch.tensor(cw, dtype=torch.float32, device=device).clamp(
            max=self.weight_cap)
        cw_t = cw_t / cw_t.sum() * K

        m = max(1, int(self.n_prototypes))

        # Prototypes initialised at class mean with small jitter for m>1
        rng_proto = np.random.default_rng(
            (self.random_state or 0) + 1)
        class_means = np.stack([
            X_tr_n[y_tr == i].mean(0) for i in range(K)])
        proto_init  = np.tile(
            class_means[:, None, :], (1, m, 1))
        if m > 1:
            jitter = rng_proto.normal(
                0, 0.05, proto_init.shape).astype(np.float32)
            proto_init = proto_init + jitter

        centroids = tt(proto_init)

        self.net_ = self._Net(centroids, K, n_feat, m=m,
                              branch_set=self.branch_set,
                              per_class_tau=self.per_class_tau,
                              use_dual_boundary=self.use_dual_boundary
                              ).to(device)

        if self.verbose:
            print(f"  device={device}  use_amp={use_amp}")
            print(f"  branch_set={self.branch_set!r}  "
                  f"active branches: {self.net_.active_branches}")

        opt = optim.Adam(self.net_.parameters(),
                         lr=self.lr, weight_decay=self.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                  opt, T_0=30, T_mult=2, eta_min=1e-5)
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        X_tr_t = tt(X_tr_n)
        y_tr_t = tt(y_tr, long=True)
        X_va_t = tt(X_va_n)
        N_tr   = len(X_tr_t)
        bs     = self.batch_size
        n_batch = max(1, N_tr // bs)

        # Cache class means once — they don't change with epochs and they
        # drive the m=1 EMA path. Big memory-bandwidth win on imbalanced
        # data (avoid re-copying the majority class every epoch).
        class_means_t = None
        if self.ema_centroids and m == 1:
            with torch.no_grad():
                class_means_t = torch.stack([
                    X_tr_t[y_tr_t == cls].mean(0) for cls in range(K)
                ])  # (K, F), on device

        val_every = 5
        best_f1, best_state, wait = -1.0, None, 0
        WARMUP = 20

        for epoch in range(self.epochs):
            # Index-on-demand shuffle: permutation tensor lives on device,
            # we slice it for each batch and gather rows from X_tr_t.
            # No full-tensor copy per epoch (was X_shuf = X_tr_t[perm]).
            perm = torch.from_numpy(rng.permutation(N_tr)).to(
                device, non_blocking=True)

            self.net_.train()
            self.net_._sep_cache = None
            for i in range(n_batch):
                idx = perm[i*bs:(i+1)*bs]
                xb  = X_tr_t[idx]
                yb  = y_tr_t[idx]

                opt.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        proba, fw, emb = self.net_(xb)
                        loss = self._loss(
                            proba, yb, cw_t, fw=fw,
                            embeddings=emb if self.supcon else None)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        self.net_.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    proba, fw, emb = self.net_(xb)
                    loss = self._loss(
                        proba, yb, cw_t, fw=fw,
                        embeddings=emb if self.supcon else None)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.net_.parameters(), 1.0)
                    opt.step()
            sch.step()

            # EMA centroid update
            if self.ema_centroids:
                self.net_.eval()
                with torch.no_grad():
                    if m == 1:
                        # Vectorised: blend all centroids against cached means.
                        # No per-class mask, no per-class allocation.
                        self.net_.centroids.data[:, 0] = (
                            self.ema_beta * self.net_.centroids.data[:, 0]
                            + (1.0 - self.ema_beta) * class_means_t)
                    else:
                        # Soft-assigned weighted mean — depends on current
                        # prototype positions, has to run per epoch.
                        for cls in range(K):
                            mask = (y_tr_t == cls)
                            if mask.sum() == 0:
                                continue
                            cls_X  = X_tr_t[mask]
                            protos = self.net_.centroids[cls]
                            _sub = cls_X
                            if len(cls_X) > 5000:
                                _idx = torch.randperm(
                                    len(cls_X), device=device)[:5000]
                                _sub = cls_X[_idx]
                            d_proto = torch.cdist(_sub, protos, p=2)
                            resp    = torch.softmax(-d_proto, dim=1)
                            for j in range(m):
                                w     = resp[:, j:j+1]
                                w_sum = w.sum().clamp(min=1e-10)
                                wm    = (w * _sub).sum(0) / w_sum
                                self.net_.centroids.data[cls, j] = (
                                    self.ema_beta       * protos[j]
                                    + (1.0 - self.ema_beta) * wm)

            # Skip validation on non-val epochs
            if (epoch + 1) % val_every != 0 and epoch < self.epochs - 1:
                if self.verbose and (epoch + 1) % 10 == 0:
                    wa = self.net_._avm_branch_weights().detach().cpu().numpy()
                    bstr = "  ".join(
                        f"{BRANCH_DISPLAY[n][:3]}:{w:.2f}"
                        for n, w in zip(self.net_.active_branches, wa))
                    print(f"  epoch {epoch+1:3d}  (val skipped)  AVM[{bstr}]")
                if wait >= self.patience:
                    if self.verbose:
                        print(f"  early stopping at epoch {epoch+1}")
                    break
                continue

            self.net_.eval()
            with torch.no_grad():
                vp, _, _ = self.net_(X_va_t)
            vf1 = self._val_score(y_va, vp.cpu().numpy().argmax(1))

            if vf1 > best_f1:
                best_f1    = vf1
                best_state = {k: v.detach().clone() for k, v
                              in self.net_.state_dict().items()}
                wait = 0
            else:
                if epoch >= WARMUP:
                    wait += 5

            if self.verbose and (epoch + 1) % 10 == 0:
                wa = self.net_._avm_branch_weights().detach().cpu().numpy()
                bstr = "  ".join(
                    f"{BRANCH_DISPLAY[n][:3]}:{w:.2f}"
                    for n, w in zip(self.net_.active_branches, wa))
                print(f"  epoch {epoch+1:3d}  val_f1={vf1:.4f}  AVM[{bstr}]")

            if wait >= self.patience:
                if self.verbose:
                    print(f"  early stopping at epoch {epoch+1}")
                break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.best_val_f1_ = best_f1
        self._n_train_    = len(X_tr)
        self._n_val_      = len(X_va)
        self.active_branches_ = list(self.net_.active_branches)

        # Extract learned parameters (host-side numpy)
        with torch.no_grad():
            self.branch_weights_ = self.net_._avm_branch_weights().cpu().numpy().copy()
            self.feat_weights_   = self.net_._feat_weights().cpu().numpy().copy()
            self.tau_            = torch.exp(self.net_.log_tau).cpu().numpy().copy()

        # ── Post-fit numpy combined-vector builders ──────────────────────────
        # All inference paths run on numpy (CPU) — FAISS and Mahalanobis live
        # there. Extract the learned scaling factors once.
        sqfw   = np.sqrt(self.feat_weights_).astype(np.float32)
        wa     = self.branch_weights_
        active = self.active_branches_
        use_dual = self.use_dual_boundary

        def _np_branch_feature(name, chunk, sqfw_local):
            eps = 1e-7
            if name == 'tac':
                a = np.arccos(np.clip(chunk, -1+eps, 1-eps))
                return np.tanh(a * 0.8) * sqfw_local
            if name == 'b':
                norm_sq = (chunk**2).sum(1, keepdims=True)
                if use_dual:
                    bnd_p = norm_sq - 2.0*chunk + 1.0
                    bnd_n = norm_sq + 2.0*chunk + 1.0
                    stacked = np.stack([bnd_p, bnd_n], axis=2)
                    return np.sqrt(
                        stacked.reshape(stacked.shape[0], -1).clip(min=eps))
                return np.sqrt((norm_sq - 2.0*chunk + 1.0).clip(min=eps))
            if name == 'cir':
                return np.sqrt(
                    np.clip(1.0 - chunk*chunk, eps, None)) * sqfw_local
            if name == 'sq':
                return (chunk * chunk) * sqfw_local
            if name == 'mtac':
                a = np.arccos(np.clip(-chunk, -1+eps, 1-eps))
                return np.tanh(a * 0.8) * sqfw_local
            raise ValueError(f"unknown branch: {name}")

        gate_sqrt = [np.float32(np.sqrt(w)) for w in wa]

        def _make_avm_combined(xn, batch_size=10_000):
            def _chunk(chunk):
                parts = [
                    gate_sqrt[i] * _np_branch_feature(name, chunk, sqfw)
                    for i, name in enumerate(active)
                ]
                return np.concatenate(parts, axis=1).astype(np.float32)
            if len(xn) <= batch_size:
                return _chunk(xn)
            return np.vstack([_chunk(xn[i:i+batch_size])
                               for i in range(0, len(xn), batch_size)])

        def _make_knn_combined(xn, batch_size=10_000):
            """KNN space: linear + shape + quadratic (3F), uniform branches."""
            def _chunk(chunk):
                eps = 1e-7
                lin = chunk * sqfw
                mu  = chunk.mean(1, keepdims=True)
                sd  = chunk.std(1, keepdims=True).clip(min=eps)
                shp = (chunk - mu) / sd
                sqd = chunk * chunk * sqfw
                return np.concatenate([lin, shp, sqd],
                                      axis=1).astype(np.float32)
            if len(xn) <= batch_size:
                return _chunk(xn)
            return np.vstack([_chunk(xn[i:i+batch_size])
                               for i in range(0, len(xn), batch_size)])

        self._make_avm_combined = _make_avm_combined
        self._make_knn_combined = _make_knn_combined

        # Extract learned prototype positions in AVM combined space
        raw_c_n     = self.net_.centroids.detach().cpu().numpy()
        init_c_n    = self.net_.centroid_anchor.cpu().numpy()
        raw_c_flat  = raw_c_n.reshape(K * m, n_feat)
        self.centroids_combined_ = _make_avm_combined(
            raw_c_flat).astype(np.float32)
        self.n_prototypes_ = m

        # Drift: mean distance each prototype moved from its anchor
        drift_flat = np.sqrt(((raw_c_n - init_c_n) ** 2).sum(-1))
        self.centroid_drift_ = drift_flat.mean(1)

        counts_tr = np.array([(y_tr == i).sum() for i in range(K)],
                              dtype=np.int32)
        self._class_counts_ = counts_tr.copy()
        if self.verbose:
            for i in range(K):
                proto_drifts = "  ".join(
                    f"p{j}={drift_flat[i,j]:.4f}" for j in range(m))
                print(f"  centroid cls{self.classes_[i]}: "
                      f"n={int(counts_tr[i])}  drift=[{proto_drifts}]")

        # Mahalanobis RDA covariance — see comments below for math
        if self.mahalanobis:
            X_tr_c = _make_avm_combined(X_tr_n)
            D      = X_tr_c.shape[1]

            S_pooled = np.zeros((D, D), dtype=np.float64)
            for i in range(K):
                cls_pts = X_tr_c[y_tr == i].astype(np.float64)
                if len(cls_pts) < 2:
                    continue
                c = cls_pts - cls_pts.mean(0, keepdims=True)
                S_pooled += (c.T @ c)
            S_pooled /= (len(X_tr_c) - K)

            self.cov_inv_  = np.zeros((K * m, D, D), dtype=np.float32)
            alpha          = float(self.mahal_alpha)
            ridge          = float(self.mahal_reg)

            # Build per-prototype soft assignments in normalised space.
            # CPU tensor — the soft-assignment math is tiny (~K*n_k*m) and
            # not worth the device round-trip for this one-shot computation.
            X_tr_n_t = torch.tensor(X_tr_n, dtype=torch.float32)

            for i in range(K):
                cls_mask = (y_tr == i)
                cls_pts  = X_tr_c[cls_mask].astype(np.float64)
                n_k      = len(cls_pts)

                if n_k < 2:
                    for j in range(m):
                        self.cov_inv_[i * m + j] = self._invert_cov(
                            S_pooled, ridge, D)
                    continue

                if m == 1:
                    c   = cls_pts - cls_pts.mean(0, keepdims=True)
                    S_k = (c.T @ c) / (n_k - 1)
                    S_blend = (1.0 - alpha) * S_k + alpha * S_pooled
                    self.cov_inv_[i] = self._invert_cov(S_blend, ridge, D)
                else:
                    cls_pts_raw = X_tr_n_t[torch.tensor(cls_mask)]
                    protos_raw  = torch.tensor(raw_c_n[i],
                                                dtype=torch.float32)
                    d_proto = torch.cdist(cls_pts_raw, protos_raw, p=2)
                    resp    = torch.softmax(-d_proto, dim=1).numpy()

                    for j in range(m):
                        w     = resp[:, j].astype(np.float64)
                        w_sum = max(w.sum(), 1e-10)
                        mu_j  = (w[:, None] * cls_pts).sum(0) / w_sum
                        c     = cls_pts - mu_j
                        S_j   = (w[:, None] * c).T @ c / w_sum
                        ess   = (w.sum() ** 2) / (w ** 2).sum().clip(min=1e-10)
                        if ess > 1:
                            S_j *= ess / (ess - 1)
                        S_blend = (1.0 - alpha) * S_j + alpha * S_pooled
                        self.cov_inv_[i * m + j] = self._invert_cov(
                            S_blend, ridge, D)

            if self.verbose:
                print(f"  Mahalanobis: per-prototype RDA "
                      f"(alpha={alpha:.2f} ridge={ridge:.0e} "
                      f"D={D} K*m={K*m})")
        else:
            self.cov_inv_ = None

        # FAISS index
        self.lambda_ = self.lam_floor
        if not _HAS_FAISS:
            if self.verbose:
                print("  faiss not installed — KNN disabled, lambda=1.0")
            self.lambda_ = 1.0
            self.index_  = None
        else:
            X_tr_c  = _make_knn_combined(X_tr_n)
            dim     = X_tr_c.shape[1]
            N_build = X_tr_c.shape[0]
            built  = False
            if self.use_ivf and N_build >= 78:
                nlist = max(1, min(
                    1000, N_build // 39,
                    int(np.sqrt(N_build))))
                try:
                    quantizer = faiss.IndexFlatL2(dim)
                    index = faiss.IndexIVFFlat(
                        quantizer, dim, nlist, faiss.METRIC_L2)
                    index.train(X_tr_c)
                    index.nprobe = min(10, nlist)
                    built = True
                    if self.verbose:
                        print(f"  FAISS IVF: nlist={nlist}"
                              f" nprobe={index.nprobe}")
                except MemoryError:
                    if self.verbose:
                        print("  FAISS IVF OOM — falling back to FlatL2")
            if not built:
                index = faiss.IndexFlatL2(dim)
                if self.verbose:
                    print(f"  FAISS Flat: {N_build} vectors dim={dim}")
            index.add(X_tr_c)
            self.index_           = index
            self.train_class_idx_ = y_tr.copy()

            if self.entropy_lambda:
                self.lambda_ = None
                if self.verbose:
                    print(f"  lambda mode=entropy  "
                          f"confident={self.lam_confident}  "
                          f"uncertain={self.lam_uncertain}")
            else:
                X_va_avm = _make_avm_combined(X_va_n)
                avm_val  = self._avm_proba(X_va_avm)

                X_va_knn = _make_knn_combined(X_va_n)
                knn_val  = self._knn_proba(X_va_knn)

                # Scalar grid search
                n_steps     = int(round((1.0 - self.lam_floor) / 0.05)) + 1
                lam_grid    = np.linspace(self.lam_floor, 1.0, n_steps)
                best_lam_f1, best_lam = -1.0, self.lam_floor
                for lam_c in lam_grid:
                    lam_c = float(lam_c)
                    p  = lam_c * avm_val + (1.0 - lam_c) * knn_val
                    f1 = self._val_score(y_va, p.argmax(1))
                    if f1 > best_lam_f1:
                        best_lam_f1 = f1
                        best_lam    = lam_c

                # Confidence-weighted gate
                lam_conf, _ = self._lambda_gate(avm_val, knn_val)
                p_conf      = lam_conf[:,None]*avm_val + (1-lam_conf[:,None])*knn_val
                conf_f1     = self._val_score(y_va, p_conf.argmax(1))

                if conf_f1 >= best_lam_f1:
                    self.lambda_ = None
                    if self.verbose:
                        print(f"  lambda: confidence gate  val_f1={conf_f1:.4f}"
                              f" (scalar={best_lam_f1:.4f})")
                else:
                    self.lambda_ = best_lam
                    if self.verbose:
                        print(f"  lambda: scalar={best_lam:.2f}"
                              f"  val_f1={best_lam_f1:.4f}"
                              f" (conf={conf_f1:.4f})")

        return self

    # ── inference helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _invert_cov(S, ridge, D):
        """Ridge-regularise and Cholesky-invert a covariance matrix.
        Falls back to pseudo-inverse on numerical failure."""
        scale = np.diag(S).mean().clip(min=1e-10)
        S_reg = (1.0 - ridge) * S + ridge * scale * np.eye(D)
        try:
            L     = np.linalg.cholesky(S_reg)
            L_inv = np.linalg.solve(L, np.eye(D))
            return (L_inv.T @ L_inv).astype(np.float32)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(S_reg).astype(np.float32)

    def _avm_proba(self, combined):
        """
        Multi-prototype AVM scoring.
        Aggregates inverse-distances per class by summing over prototypes.
        Per-prototype Mahalanobis when n_prototypes>1.
        """
        K = len(self.classes_)
        m = self.n_prototypes_
        N = combined.shape[0]

        if self.cov_inv_ is not None:
            d_flat = np.zeros((N, K * m), dtype=np.float64)
            comb64 = combined.astype(np.float64)
            for km in range(K * m):
                diff = comb64 - self.centroids_combined_[km].astype(np.float64)
                tmp  = diff @ self.cov_inv_[km].astype(np.float64)
                d_flat[:, km] = np.sqrt(
                    np.maximum((tmp * diff).sum(1), 0.0))
            d_flat = d_flat.astype(np.float32)
        else:
            a_sq   = (combined ** 2).sum(1, keepdims=True)
            b_sq   = (self.centroids_combined_ ** 2).sum(1)
            ab     = combined @ self.centroids_combined_.T
            d_flat = np.sqrt(np.maximum(
                a_sq + b_sq[None, :] - 2.0 * ab, 0.0)).astype(np.float32)

        d_km = d_flat.reshape(N, K, m)
        raw  = (1.0 / (d_km + 1e-10)).sum(2)
        return raw / raw.sum(1, keepdims=True)

    def _knn_proba(self, combined, search_batch=50_000):
        if self.index_ is None:
            raise RuntimeError("FAISS index not built.")
        n_test = combined.shape[0]
        K      = len(self.classes_)
        proba  = np.zeros((n_test, K), dtype=np.float32)
        for start in range(0, n_test, search_batch):
            end       = min(start + search_batch, n_test)
            dist, idx = self.index_.search(combined[start:end], self.k)
            w = 1.0 / (dist + 1e-10)
            w = w / w.sum(1, keepdims=True)
            labels = self.train_class_idx_[idx]
            B = end - start
            batch_p = np.zeros((B, K), dtype=np.float32)
            for ki in range(self.k):
                batch_p[np.arange(B), labels[:, ki]] += w[:, ki]
            proba[start:end] = batch_p
        return proba

    def _lambda_gate(self, avm, knn):
        """Confidence-weighted gate: lambda = conf_AVM / (conf_AVM + conf_KNN)."""
        K     = len(self.classes_)
        log_K = np.log(K + 1e-10)
        H_avm = -(avm * np.log(avm.clip(1e-10))).sum(1) / log_K
        H_knn = -(knn * np.log(knn.clip(1e-10))).sum(1) / log_K
        c_avm = 1.0 - H_avm
        c_knn = 1.0 - H_knn
        lam   = (c_avm / (c_avm + c_knn + 1e-10)).clip(
            self.lam_uncertain, self.lam_confident)
        overlap  = np.minimum(avm, knn).sum(1)
        disagree = c_avm * c_knn * (1 - overlap)
        return lam, disagree

    def predict_proba(self, X):
        check_is_fitted(self, 'centroids_combined_')
        X      = np.asarray(X, dtype=np.float32)
        Xn     = self._norm_apply(X)
        Xn_avm = self._make_avm_combined(Xn)
        avm    = self._avm_proba(Xn_avm)

        if self.index_ is None or (
                self.lambda_ is not None and self.lambda_ >= 1.0):
            return avm

        knn = self._knn_proba(self._make_knn_combined(Xn))

        if self.entropy_lambda:
            K   = len(self.classes_)
            H   = -(avm * np.log(avm.clip(1e-10))).sum(1) / np.log(K)
            lam = (self.lam_confident
                   - (self.lam_confident - self.lam_uncertain) * H)
            lam = lam.clip(self.lam_uncertain, self.lam_confident)
            return lam[:,None]*avm + (1-lam[:,None])*knn

        elif self.lambda_ is None:
            lam, _ = self._lambda_gate(avm, knn)
            return lam[:,None]*avm + (1-lam[:,None])*knn

        else:
            return self.lambda_*avm + (1.0-self.lambda_)*knn

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_with_uncertainty(self, X):
        """
        Returns (predictions, proba, disagreement_scores).
        Gate weights AVM vs KNN by per-sample confidence.
        """
        check_is_fitted(self, 'centroids_combined_')
        X      = np.asarray(X, dtype=np.float32)
        Xn     = self._norm_apply(X)
        Xn_avm = self._make_avm_combined(Xn)
        avm    = self._avm_proba(Xn_avm)

        if self.index_ is None:
            zero_d = np.zeros(len(X), dtype=np.float32)
            return self.classes_[avm.argmax(1)], avm, zero_d

        knn = self._knn_proba(self._make_knn_combined(Xn))
        lam, disagree = self._lambda_gate(avm, knn)
        proba         = lam[:,None]*avm + (1-lam[:,None])*knn
        return self.classes_[proba.argmax(1)], proba, disagree

    def class_distribution(self, as_dict=False):
        check_is_fitted(self, 'classes_')
        counts = getattr(self, '_class_counts_', None)
        if counts is None:
            return {} if as_dict else "class counts not available"
        total = counts.sum()
        if as_dict:
            return {c: float(n / total * 100)
                    for c, n in zip(self.classes_, counts)}
        return "  ".join(
            f"{c}: {n/total*100:.0f}%"
            for c, n in zip(self.classes_, counts))

    def summary(self, feature_names=None):
        """Print a human-readable model summary after fitting."""
        check_is_fitted(self, 'branch_weights_')
        names = (list(feature_names) if feature_names is not None
                 else [f"f{i}" for i in range(len(self.feat_weights_))])
        bw  = self.branch_weights_
        ab  = self.active_branches_
        lam = getattr(self, 'lambda_', self.lam_floor)
        tau = self.tau_
        tau_str = (f"{float(tau):.3f}" if np.ndim(tau) == 0 or np.size(tau) == 1
                   else "  ".join(f"cls{i}:{v:.3f}" for i, v in enumerate(tau)))
        lam_str = ("[confidence gate]" if lam is None else
                   "[entropy gate]"    if self.entropy_lambda else
                   f"{lam:.3f}")
        K     = len(self.classes_)
        n_tr  = getattr(self, '_n_train_', '?')
        n_va  = getattr(self, '_n_val_',   '?')
        dev   = getattr(self, 'device_', '?')
        dist  = self.class_distribution()
        lines = [
            f"  Classes ({K}): {dist}",
            f"  Train / val  : {n_tr} / {n_va}",
            f"  Device       : {dev}",
            "",
            f"  Lambda  : {lam_str}",
            f"  Tau     : {tau_str}",
            f"  Val F1  : {getattr(self, 'best_val_f1_', float('nan')):.4f}",
            "",
            f"  Mahalanobis : {'full RDA' if self.mahalanobis else 'off'}"
            f"  (alpha={self.mahal_alpha}  ridge={self.mahal_reg:.0e})",
            f"  Ordinal EMD : {self.ordinal}  (weight={self.ordinal_weight})",
            f"  SupCon      : {self.supcon}"
            f"  (weight={self.supcon_weight}  temp={self.supcon_temp})",
            f"  EMA         : {self.ema_centroids}  (beta={self.ema_beta})",
            f"  Dual bnd    : {self.use_dual_boundary}",
            f"  Branch set  : {self.branch_set}  ({len(ab)} branches)",
            f"  n_prototypes: {getattr(self, 'n_prototypes_', self.n_prototypes)}"
            f"  per_class_τ: {self.per_class_tau}",
            f"  ortho_reg   : {self.ortho_reg}",
            "",
            f"  AVM branches (learned):",
        ]
        order = np.argsort(bw)[::-1]
        for i in order:
            lines.append(
                f"    {BRANCH_DISPLAY[ab[i]]:<12}: {bw[i]:.4f}")
        lines += [
            f"  KNN voter   : fixed-feature FAISS index (no learned weights)",
            "",
            f"  Feature weights (top 5):",
        ]
        order = np.argsort(self.feat_weights_)[::-1]
        for i in order[:5]:
            lines.append(f"    {names[i]:<28} {self.feat_weights_[i]:.4f}")
        lines.append(f"  Feature weights (bottom 2):")
        for i in order[-2:][::-1]:
            lines.append(f"    {names[i]:<28} {self.feat_weights_[i]:.4f}")
        if hasattr(self, 'centroid_drift_') and self.centroid_drift_ is not None:
            lines.append("")
            lines.append("  Centroid drift:")
            for cls, v in zip(self.classes_, self.centroid_drift_):
                bar = "█" * int(v * 40)
                lines.append(f"    class {cls}: {v:.4f}  {bar}")
        if lam is not None and not self.entropy_lambda:
            lines.append("")
            lines.append(f"  Lambda search: floor={self.lam_floor}"
                         f"  chosen={lam_str}")
        return "\n".join(lines)
