"""
LearningAVNN
============
Geometric tabular classifier combining a trained Angular Vector Machine
(AVM) with a per-class FAISS-accelerated KNN voter, blended at inference.

Architecture
------------
  Input → MinMax normalise [-1,1]
       → AVM: multi-branch combined vector (configurable via branch_set)
            → Mahalanobis inverse-distance to K×m learnable prototypes
              → AVM probability  (trained end-to-end)
       → KNN: linear + shape + quadratic combined vector
            → K separate FAISS indices, one per class
            → per-class inverse-distance score + novelty signal
              → KNN probability + novelty (fixed features, post-fit)
       → predict: λ·AVM + (1-λ)·KNN
                  λ uses confidence gate (AVM entropy + KNN entropy +
                  KNN novelty) to weight per-sample, or scalar tuned on val.

The KNN voter is class-conditional: each test point queries every class's
own index for its k nearest in-class neighbors. This prevents the
majority class from drowning out minorities in the nearest-neighbor pool,
and provides a natural anomaly signal — a test point that's far from
every class's manifold is likely novel.

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
    sq      x²                          symmetric, peaks at x=±1
    mtac    tanh(arccos(-x)·0.8)        mirror of tac — monotone INCREASING

KNN Voter  (fixed — local structure / per-class neighbourhood similarity)
-------------------------------------------------------------------------
  linear       x·√fw           signed magnitude, direct local proximity
  shape        (x-μ)/σ         within-sample relative profile fingerprint
  quadratic    x²·√fw          nonlinear extremeness

  Per-class indices: for each class k, build a FAISS index over the
  class-k training points. At inference, search each class's index
  separately for k nearest in-class neighbours. The class-k score is
  the mean inverse-distance of those neighbours; per-class scores are
  normalised into a probability distribution.

  Novelty signal: the linear-L2 distance from the test point to its
  nearest neighbour in any class, normalised by a per-class typical
  intra-class distance computed during fit. Novelty ≈ 1 means the test
  point is at a typical training distance; novelty >> 1 means the point
  is far from every class manifold and the KNN view should be distrusted.

Performance
-----------
  device='auto'      Picks 'cuda' if available, else 'cpu'.
  Mixed precision    Automatic on CUDA via torch.amp.autocast.
  Index-on-demand    Shuffle indices once per epoch, slice X_tr_t inside
                     the batch loop. No full-tensor copy per epoch.
  Cached class means EMA path with m=1 reuses class means computed once.

Imbalance Handling
------------------
  SupCon loss     — pulls same-class AVM embeddings together per batch
  Per-class tau   — each class learns its own scoring sharpness
  Mahalanobis RDA — full per-class covariance at inference (α-blended)
  Class weights   — balanced NLL capped at weight_cap
  Ordinal EMD     — Earth Mover's Distance for ordered class structures
  EMA centroids   — exponential moving average stabilises minority prototypes
  val_macro_bias  — early stopping biased toward macro F1 over accuracy
  Class-cond KNN  — each class gets its own neighbour pool, no majority drown

Key Parameters
--------------
  k              int    KNN neighbours per class
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
BRANCH_SETS = {
    'default':    ['tac', 'b',    'cir'],
    'extended':   ['tac', 'b',    'cir', 'sq', 'mtac'],
    'tac_family': ['tac', 'mtac', 'sq'],
}

BRANCH_INITS = {
    'tac':  0.5,
    'b':    0.5,
    'cir':  0.0,
    'sq':   0.0,
    'mtac': 0.0,
}

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


def _build_faiss_index(X, dim, use_ivf=True):
    """
    Build a FAISS L2 index over X. Uses IVF for large collections, Flat
    otherwise. Returns the index.
    """
    n = X.shape[0]
    if use_ivf and n >= 78:
        nlist = max(1, min(1000, n // 39, int(np.sqrt(n))))
        try:
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_L2)
            index.train(X)
            index.nprobe = min(10, nlist)
            index.add(X)
            return index, ('IVF', nlist)
        except MemoryError:
            pass
    index = faiss.IndexFlatL2(dim)
    index.add(X)
    return index, ('Flat', None)


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

            self.raw_w = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(BRANCH_INITS[name]))
                for name in self.active_branches
            })

            self.raw_feat_w = nn.Parameter(torch.zeros(n_feat))

        def _avm_branch_weights(self):
            raw   = torch.stack(
                [self.raw_w[name] for name in self.active_branches])
            gates = torch.sigmoid(raw)
            return gates / (gates.sum() + 1e-10)

        def _feat_weights(self):
            w = torch.softmax(self.raw_feat_w, dim=0)
            return w * self.n_feat

        @staticmethod
        def _boundary_dists(x, eps, dual=False):
            norm_sq     = (x**2).sum(dim=1, keepdim=True)
            dist_sq_pos = (norm_sq - 2.0*x + 1.0).clamp(min=eps)
            if not dual:
                return torch.sqrt(dist_sq_pos)
            dist_sq_neg = (norm_sq + 2.0*x + 1.0).clamp(min=eps)
            stacked = torch.stack([dist_sq_pos, dist_sq_neg], dim=2)
            return torch.sqrt(stacked.reshape(stacked.shape[0], -1))

        def _branch_feature(self, name, x, wf, sqwf):
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
            sqwf = wf.sqrt()
            wa   = self._avm_branch_weights()
            parts = []
            for i, name in enumerate(self.active_branches):
                feat = self._branch_feature(name, x, wf, sqwf)
                parts.append(wa[i].sqrt() * feat)
            return torch.cat(parts, dim=1)

        def forward(self, x):
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
        N, K   = probs.shape
        cdf_p  = torch.cumsum(probs, dim=1)[:, :-1]
        k_idx  = torch.arange(K - 1, device=targets.device)
        cdf_y  = (targets.unsqueeze(1) > k_idx.unsqueeze(0)).float()
        return torch.abs(cdf_p - cdf_y).mean()

    def _supcon_loss(self, embeddings, targets):
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
        smooth_term = -torch.log(probs.clamp(1e-10)).mean(dim=1).mean()
        ce  = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth_term

        _fw = fw if fw is not None else self.net_._feat_weights()
        fw_reg = self.feat_weight_reg * (_fw - 1.0).pow(2).mean()

        c_anchor = self.centroid_reg * (
            self.net_.centroids - self.net_.centroid_anchor
        ).pow(2).mean()

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

        rng_proto = np.random.default_rng((self.random_state or 0) + 1)
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

        class_means_t = None
        if self.ema_centroids and m == 1:
            with torch.no_grad():
                class_means_t = torch.stack([
                    X_tr_t[y_tr_t == cls].mean(0) for cls in range(K)
                ])

        val_every = 5
        best_f1, best_state, wait = -1.0, None, 0
        WARMUP = 20

        for epoch in range(self.epochs):
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

            if self.ema_centroids:
                self.net_.eval()
                with torch.no_grad():
                    if m == 1:
                        self.net_.centroids.data[:, 0] = (
                            self.ema_beta * self.net_.centroids.data[:, 0]
                            + (1.0 - self.ema_beta) * class_means_t)
                    else:
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

        with torch.no_grad():
            self.branch_weights_ = self.net_._avm_branch_weights().cpu().numpy().copy()
            self.feat_weights_   = self.net_._feat_weights().cpu().numpy().copy()
            self.tau_            = torch.exp(self.net_.log_tau).cpu().numpy().copy()

        # ── Post-fit numpy combined-vector builders ──────────────────────────
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

        raw_c_n     = self.net_.centroids.detach().cpu().numpy()
        init_c_n    = self.net_.centroid_anchor.cpu().numpy()
        raw_c_flat  = raw_c_n.reshape(K * m, n_feat)
        self.centroids_combined_ = _make_avm_combined(
            raw_c_flat).astype(np.float32)
        self.n_prototypes_ = m

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

        # Mahalanobis RDA covariance — see comments in earlier versions
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

        # ── Class-conditional KNN voter ──────────────────────────────────────
        # Build K separate FAISS indices, one per class, over each class's
        # training points in KNN combined space. Each test point will query
        # every index for its k nearest in-class neighbours. Per-class
        # inverse-distance score → probability. Per-class minimum distance
        # → novelty signal.
        self.lambda_ = self.lam_floor
        self.knn_indexes_ = None
        self.knn_index_kinds_ = None
        self.dist_scale_per_class_ = None
        self.knn_class_counts_ = None

        if not _HAS_FAISS:
            if self.verbose:
                print("  faiss not installed — KNN disabled, lambda=1.0")
            self.lambda_ = 1.0
        else:
            X_tr_c = _make_knn_combined(X_tr_n)
            dim    = X_tr_c.shape[1]

            self.knn_indexes_         = []
            self.knn_index_kinds_     = []
            self.dist_scale_per_class_ = []
            self.knn_class_counts_     = []

            for cls_idx in range(K):
                cls_mask = (y_tr == cls_idx)
                X_cls    = np.ascontiguousarray(X_tr_c[cls_mask])
                n_cls    = len(X_cls)
                self.knn_class_counts_.append(int(n_cls))

                if n_cls == 0:
                    # Degenerate — class has no training samples in this fold.
                    # Build a dummy 1-point index that will always return
                    # large distances.
                    dummy = np.zeros((1, dim), dtype=np.float32)
                    idx, kind = _build_faiss_index(
                        dummy, dim, use_ivf=False)
                    self.knn_indexes_.append(idx)
                    self.knn_index_kinds_.append(('Empty', None))
                    self.dist_scale_per_class_.append(1.0)
                    continue

                idx, kind = _build_faiss_index(
                    X_cls, dim, use_ivf=self.use_ivf)
                self.knn_indexes_.append(idx)
                self.knn_index_kinds_.append(kind)

                # Per-class typical intra-class distance — calibrates the
                # novelty signal. We self-search a subsample of the class
                # for k=2 (self + nearest other), take the linear-distance
                # to the nearest neighbour (skip self), median across the
                # subsample. Linear (sqrt) distance for interpretability,
                # FAISS returns squared L2.
                if n_cls >= 2:
                    sub_n = min(5000, n_cls)
                    if sub_n < n_cls:
                        sub_idx = rng.choice(n_cls, sub_n, replace=False)
                        X_sub = np.ascontiguousarray(X_cls[sub_idx])
                    else:
                        X_sub = X_cls
                    D_self, _ = idx.search(X_sub, min(2, n_cls))
                    # D_self[:, 0] is to self (0 for exact match);
                    # D_self[:, 1] is to nearest other (squared L2).
                    if D_self.shape[1] >= 2:
                        nn_sq = D_self[:, 1]
                    else:
                        nn_sq = D_self[:, 0]
                    nn_lin = np.sqrt(np.maximum(nn_sq, 0.0))
                    scale  = float(np.median(nn_lin))
                    self.dist_scale_per_class_.append(
                        max(scale, 1e-6))
                else:
                    self.dist_scale_per_class_.append(1.0)

            if self.verbose:
                kind_summary = " ".join(
                    f"cls{self.classes_[i]}:{self.knn_index_kinds_[i][0]}({self.knn_class_counts_[i]})"
                    for i in range(K))
                print(f"  KNN per-class indices: {kind_summary}")
                scales = " ".join(
                    f"cls{self.classes_[i]}:{self.dist_scale_per_class_[i]:.4f}"
                    for i in range(K))
                print(f"  KNN dist scales (typical intra-class): {scales}")

            # Lambda selection on val set
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
                knn_val, novelty_val = self._knn_proba(X_va_knn)

                # Option A: scalar grid search (ignores novelty — it's just
                # a fixed AVM/KNN ratio)
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

                # Option B: confidence-weighted gate (uses novelty)
                lam_conf, _ = self._lambda_gate(
                    avm_val, knn_val, knn_novelty=novelty_val)
                p_conf  = lam_conf[:,None]*avm_val + (1-lam_conf[:,None])*knn_val
                conf_f1 = self._val_score(y_va, p_conf.argmax(1))

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
        scale = np.diag(S).mean().clip(min=1e-10)
        S_reg = (1.0 - ridge) * S + ridge * scale * np.eye(D)
        try:
            L     = np.linalg.cholesky(S_reg)
            L_inv = np.linalg.solve(L, np.eye(D))
            return (L_inv.T @ L_inv).astype(np.float32)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(S_reg).astype(np.float32)

    def _avm_proba(self, combined):
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
        """
        Class-conditional KNN scoring.

        For each test point and each class k, search class-k's FAISS index
        for the k nearest in-class neighbours. The class-k score is the
        mean inverse-distance to those neighbours; per-class scores are
        normalised into a probability distribution.

        Returns
        -------
        proba : (N, K) per-class probability from inverse-distance voting
        novelty : (N,) novelty signal — distance to nearest in-class
                  neighbour across all classes, normalised by the global
                  median of per-class typical intra-class distances.
                  novelty ≈ 1 means "typical training distance"; novelty
                  >> 1 means "far from every class manifold."
        """
        if self.knn_indexes_ is None:
            raise RuntimeError("FAISS indices not built.")
        n_test = combined.shape[0]
        K      = len(self.classes_)

        # Score per class: mean inverse-distance over class-k neighbours
        proba_raw    = np.zeros((n_test, K), dtype=np.float64)
        # Track nearest-neighbour distance per class for novelty
        nearest_sq   = np.full((n_test, K), np.inf, dtype=np.float64)

        combined_c = np.ascontiguousarray(combined)

        for cls_idx in range(K):
            idx_obj = self.knn_indexes_[cls_idx]
            n_in_idx = idx_obj.ntotal
            if n_in_idx == 0:
                # Empty class (shouldn't happen with stratified split, but
                # be defensive). Score stays 0; nearest stays inf.
                continue

            k_search = min(self.k, n_in_idx)

            for start in range(0, n_test, search_batch):
                end = min(start + search_batch, n_test)
                dist, _ = idx_obj.search(combined_c[start:end], k_search)
                # FAISS returns squared L2 for IndexFlatL2 / IndexIVFFlat
                # Inverse-distance vote (per-neighbour weights, then mean)
                w = 1.0 / (dist + 1e-10)
                proba_raw[start:end, cls_idx] = w.mean(1)
                # Track nearest (smallest squared distance)
                nearest_sq[start:end, cls_idx] = dist[:, 0]

        # Normalise into probabilities. If a row is all-zero (no class has
        # any training data — pathological case) fall back to uniform.
        row_sums = proba_raw.sum(1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        proba    = (proba_raw / row_sums).astype(np.float32)

        # Novelty: linear distance to the single nearest in-class neighbour
        # (across all classes), normalised by the global median of per-class
        # typical intra-class distances. The global median is used so the
        # novelty value is comparable across classes — a per-class scale
        # would let the model declare every test point "typical for some
        # class" which defeats the purpose.
        nearest_lin = np.sqrt(np.maximum(nearest_sq.min(1), 0.0))
        global_scale = float(np.median(self.dist_scale_per_class_))
        novelty = (nearest_lin / max(global_scale, 1e-10)).astype(np.float32)

        return proba, novelty

    def _lambda_gate(self, avm, knn, knn_novelty=None):
        """
        Confidence-weighted gate: lambda = conf_AVM / (conf_AVM + conf_KNN).

        Confidence sources:
          AVM: 1 - normalised entropy of AVM probabilities
          KNN: (1 - normalised entropy of KNN probabilities)
               × novelty_factor (if novelty signal provided)

        Novelty factor:  1 / (1 + max(novelty - 1, 0)^2)
          novelty=1 → factor=1.0   (typical training distance)
          novelty=2 → factor=0.5   (twice as far as typical)
          novelty=3 → factor=0.2
          novelty=5 → factor=0.06
        This means a test point that's far from every class manifold has
        its KNN confidence collapsed, so the gate leans on AVM (or, if
        AVM is also uncertain, the lambda gets pulled to the lam_uncertain
        floor and the prediction is genuinely a coin flip — which is the
        honest answer).
        """
        K     = len(self.classes_)
        log_K = np.log(K + 1e-10)
        H_avm = -(avm * np.log(avm.clip(1e-10))).sum(1) / log_K
        H_knn = -(knn * np.log(knn.clip(1e-10))).sum(1) / log_K
        c_avm = 1.0 - H_avm
        c_knn = 1.0 - H_knn

        if knn_novelty is not None:
            # Inverse-quadratic falloff above novelty=1
            excess = np.maximum(knn_novelty - 1.0, 0.0)
            novelty_factor = 1.0 / (1.0 + excess * excess)
            c_knn = c_knn * novelty_factor

        lam = (c_avm / (c_avm + c_knn + 1e-10)).clip(
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

        if self.knn_indexes_ is None or (
                self.lambda_ is not None and self.lambda_ >= 1.0):
            return avm

        knn, novelty = self._knn_proba(self._make_knn_combined(Xn))

        if self.entropy_lambda:
            K   = len(self.classes_)
            H   = -(avm * np.log(avm.clip(1e-10))).sum(1) / np.log(K)
            lam = (self.lam_confident
                   - (self.lam_confident - self.lam_uncertain) * H)
            lam = lam.clip(self.lam_uncertain, self.lam_confident)
            return lam[:,None]*avm + (1-lam[:,None])*knn

        elif self.lambda_ is None:
            lam, _ = self._lambda_gate(avm, knn, knn_novelty=novelty)
            return lam[:,None]*avm + (1-lam[:,None])*knn

        else:
            return self.lambda_*avm + (1.0-self.lambda_)*knn

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_with_uncertainty(self, X):
        """
        Returns (predictions, proba, disagreement, novelty).
        Gate weights AVM vs KNN by per-sample confidence including novelty.
        """
        check_is_fitted(self, 'centroids_combined_')
        X      = np.asarray(X, dtype=np.float32)
        Xn     = self._norm_apply(X)
        Xn_avm = self._make_avm_combined(Xn)
        avm    = self._avm_proba(Xn_avm)

        if self.knn_indexes_ is None:
            zero_d = np.zeros(len(X), dtype=np.float32)
            zero_n = np.zeros(len(X), dtype=np.float32)
            return self.classes_[avm.argmax(1)], avm, zero_d, zero_n

        knn, novelty  = self._knn_proba(self._make_knn_combined(Xn))
        lam, disagree = self._lambda_gate(avm, knn, knn_novelty=novelty)
        proba         = lam[:,None]*avm + (1-lam[:,None])*knn
        return self.classes_[proba.argmax(1)], proba, disagree, novelty

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
            f"  KNN voter   : class-conditional ({K} per-class indices)",
        ]
        # Per-class KNN info if available
        if getattr(self, 'knn_class_counts_', None) is not None:
            lines.append("  Per-class KNN:")
            for ci, cls in enumerate(self.classes_):
                kind = self.knn_index_kinds_[ci][0] if self.knn_index_kinds_ else '?'
                cnt  = self.knn_class_counts_[ci]
                sc   = self.dist_scale_per_class_[ci]
                lines.append(
                    f"    class {cls}: n={cnt}  index={kind}  scale={sc:.4f}")
        lines += [
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
