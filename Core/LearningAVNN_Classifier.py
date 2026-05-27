"""
LearningAVNN
============
Geometric tabular classifier combining Angular Vector Machines (AVM)
with FAISS-accelerated KNN and a suite of learned geometric corrections.

Architecture Overview
---------------------
  Input → MinMax normalise [-1,1] → Six-branch combined vector
       → AVM: inverse-distance to K×m learnable prototypes
       → KNN: FAISS nearest-neighbour inverse-distance vote
       → λ·AVM + (1-λ)·KNN  (λ auto-tuned post-training)

Six Branches
------------
  0  linear      x_f · √fw               direction + magnitude, linear
  1  circular    √(1-x²) · √fw           symmetric extremeness
  2  boundary    √(‖x‖²-2x+1)            cross-feature coupling via ‖x‖²
  3  shape       (x-μ)/σ  per-sample     within-sample relative profile
  4  quadratic   x² · √fw                nonlinear extremeness
  5  tanh_arccos tanh(arccos(x)·0.8)·√fw monotone nonlinear, center-amplified

  Branches use independent normalised-sigmoid gates (not softmax) so
  boundary and tanh_arccos can simultaneously be strong without
  zero-sum competition. tac and boundary initialised with positive
  bias (~25% head start over the other four branches).

Triangle Scoring (optional, use_triangle=True)
----------------------------------------------
  For each feature pair (a,b) and each class prototype μ_k:
    cross_{ab,k} = x_a·μ_kb - x_b·μ_ka   (2D wedge product)

  This is the signed area of the triangle formed by the origin,
  the sample's (x_a, x_b) sub-vector, and the centroid's (μ_ka, μ_kb).
  Zero when the sample's feature ratio matches the centroid's ratio —
  measures PROFILE SIMILARITY, not just proximity.

  Captures multiplicative feature interactions ("high alcohol AND
  low volatile acidity") that axis-aligned branches cannot represent.
  Blended with geometric AVM distance via tri_weight parameter.

Learnable Prototypes
--------------------
  Each class has m learnable prototype positions (nn.Parameter, shape K×m×F).
  n_prototypes=1 reduces to a single centroid per class (original behaviour).
  n_prototypes>1 allows multimodal class representations — useful when a
  class (e.g. "medium quality" wines) spans multiple sub-clusters.

  centroid_reg  — anchors prototypes near initial class means
  centroid_sep  — pushes inter-class prototypes apart
  intra_sep     — pushes intra-class prototypes apart (forces spread)
  EMA update    — soft-assignment exponential moving average per epoch

Imbalance Handling
------------------
  Supervised Contrastive Loss (SupCon) — pulls same-class embeddings
    together; effective when ≥2 rare samples appear per batch.
  Per-class tau  — minority class centroids learn softer scoring.
  Mahalanobis    — full RDA covariance (diagonal or full, α-blended).
  Class weights  — balanced NLL with weight_cap to prevent extremes.
  Ordinal EMD    — Earth Mover's Distance for ordered classes (K>2).

Key Parameters
--------------
  n_prototypes   int    prototypes per class (1=single centroid)
  use_triangle   bool   enable triangle cross-product scoring
  tri_weight     float  blend: 0=pure AVM distance, 1=pure triangle
  mahal_alpha    float  RDA blend: 0=QDA, 1=LDA, 0.5=balanced
  lam_floor      float  minimum AVM weight in AVM/KNN blend
  entropy_lambda bool   per-sample lambda via AVM entropy gate
  supcon         bool   supervised contrastive loss
  ema_centroids  bool   EMA centroid stabilisation
  ortho_reg      float  Gram matrix branch decorrelation penalty
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


class LearningAVNN(BaseEstimator, ClassifierMixin):

    def __init__(self, k=5, lr=1e-3, epochs=300, batch_size=256,
                 patience=60, val_fraction=0.15, label_smoothing=0.05,
                 weight_cap=1.5, weight_decay=1e-4, feat_weight_reg=0.05,
                 centroid_reg=0.1, centroid_sep=0.05,
                 n_prototypes=1, intra_sep=0.05,
                 use_triangle=False, tri_weight=0.5,
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
        self.use_triangle        = use_triangle
        self.tri_weight          = tri_weight
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
                     per_class_tau=False, use_triangle=False,
                     eps=1e-7):
            super().__init__()
            self.K, self.m, self.n_feat, self.eps = K, m, n_feat, eps
            self.per_class_tau    = per_class_tau
            self.use_triangle     = use_triangle
            self._sep_cache       = None

            self.centroids = nn.Parameter(centroids.clone())
            self.register_buffer("centroid_anchor", centroids.clone())

            if per_class_tau:
                self.log_tau = nn.Parameter(torch.full((K,), -0.5))
            else:
                self.log_tau = nn.Parameter(torch.tensor(-0.5))

            if use_triangle:
                n_pairs = n_feat * (n_feat - 1) // 2
                self.log_tau_tri = nn.Parameter(torch.tensor(-0.5))
                self.log_w_pairs = nn.Parameter(torch.zeros(n_pairs))


            # Six branch weights — independent sigmoid gates, normalised.
            # tac and boundary biased positive (~25% head start).
            self.raw_w_lin  = nn.Parameter(torch.tensor(0.0))
            self.raw_w_cir  = nn.Parameter(torch.tensor(0.0))
            self.raw_w_b    = nn.Parameter(torch.tensor(0.5))
            self.raw_w_s    = nn.Parameter(torch.tensor(0.0))
            self.raw_w_sq   = nn.Parameter(torch.tensor(0.0))
            self.raw_w_tac  = nn.Parameter(torch.tensor(0.5))
            self.raw_feat_w = nn.Parameter(torch.zeros(n_feat))



        def _branch_weights(self):
            """
            6 independent sigmoid gates, normalised to sum to 1.
            tac and boundary have positive init bias (~25% head start).
            """
            raw   = torch.stack([
                self.raw_w_lin, self.raw_w_cir, self.raw_w_b,
                self.raw_w_s,   self.raw_w_sq,  self.raw_w_tac])
            gates = torch.sigmoid(raw)
            return gates / (gates.sum() + 1e-10)

        def _feat_weights(self):
            w = torch.softmax(self.raw_feat_w, dim=0)
            return w * self.n_feat   # uniform = 1.0 each

        @staticmethod
        def _boundary_dists(x, eps):
            """
            Distances from each point to the F positive face-centre anchors.

            For feature f, the positive anchor is e_f (unit vector in direction f):
              ||x - e_f||^2 = ||x||^2 - 2*x_f + 1

            Output shape: (N, F)

            Why positive anchors only (not 2F):
              The negative anchors (dist to -e_f) are dist_pos evaluated at -x_f.
              phi_dir = arccos(x_f) already encodes which boundary is nearer and
              phi_mag = arccos(|x_f|) encodes distance to the nearest boundary.
              The unique value of the boundary branch is the ||x||^2 cross-feature
              coupling — present in positive anchors alone. Dropping negative anchors
              halves the output dimension (F not 2F), reducing concat and sqrt cost
              by ~43% with negligible geometric information loss.
            """
            norm_sq = (x ** 2).sum(dim=1, keepdim=True)    # (N, 1)
            dist_sq = (norm_sq - 2.0 * x + 1.0).clamp(min=eps)  # (N, F)
            return torch.sqrt(dist_sq)                     # (N, F)

        def _combined_features(self, x, wb, wf):
            """
            6-branch combined vector.
            Branch 0 — linear      x·√fw              direction + magnitude
            Branch 1 — circular    √(1-x²)·√fw        symmetric extremeness
            Branch 2 — boundary    √(‖x‖²-2x+1)       cross-feature coupling
            Branch 3 — shape       z-score             within-sample profile
            Branch 4 — quadratic   x²·√fw              nonlinear extremeness
            Branch 5 — tanh_arccos tanh(arccos(x)·0.8)·√fw  monotone nonlinear

            Note: linear/quadratic/circular weights cluster near-uniform on
            clean data but their dimensions still enrich the KNN combined
            space — removing them degrades holdout performance even when
            gradient cannot differentiate them.
            """
            eps  = self.eps
            sqwf = wf.sqrt()

            lin  = x * sqwf
            cir  = torch.sqrt(torch.clamp(1.0 - x * x, min=eps)) * sqwf
            sqd  = x * x * sqwf

            norm_sq = (x ** 2).sum(dim=1, keepdim=True)
            bound   = torch.sqrt((norm_sq - 2.0 * x + 1.0).clamp(min=eps))

            mu  = x.mean(1, keepdim=True)
            sd  = x.std(1, keepdim=True).clamp(min=eps)
            shp = (x - mu) / sd

            a   = torch.acos(torch.clamp(x, -1 + eps, 1 - eps))
            tac = torch.tanh(a * 0.8) * sqwf

            return torch.cat([
                wb[0].sqrt() * lin,
                wb[1].sqrt() * cir,
                wb[2].sqrt() * bound,
                wb[3].sqrt() * shp,
                wb[4].sqrt() * sqd,
                wb[5].sqrt() * tac,
            ], dim=1)

        def _triangle_score(self, x, proto_flat):
            """
            Deviation triangle cross-product scoring.

            Computes cross-products in deviation space — both sample and
            centroid are centred by their own feature means before the
            wedge product:

              dx_a  = x_a  - mean(x)        sample deviation for feature a
              dμ_ka = μ_ka - mean(μ_k)      centroid deviation for feature a

              cross_{ab,k} = dx_a * dμ_kb - dx_b * dμ_ka

            Zero when the sample's RELATIVE PROFILE (which features are
            above/below its own mean) matches the centroid's relative profile
            for this pair — regardless of absolute scale differences.

            Strictly more expressive than raw ratio comparison:
              raw:       x_a/x_b  vs  μ_ka/μ_kb
              deviation: (x_a-x̄)/(x_b-x̄)  vs  (μ_ka-μ̄_k)/(μ_kb-μ̄_k)

            A sample matching the centroid's profile at a different absolute
            scale now correctly yields near-zero cross-product.
            Complements the shape branch (z-score) by capturing the same
            deviation concept at the pairwise interaction level.
            """
            F   = self.n_feat
            eps = self.eps

            pairs_a, pairs_b = [], []
            for a in range(F):
                for b in range(a + 1, F):
                    pairs_a.append(a)
                    pairs_b.append(b)
            pa = torch.tensor(pairs_a, device=x.device)
            pb = torch.tensor(pairs_b, device=x.device)

            w_pairs = torch.nn.functional.softplus(self.log_w_pairs)  # (P,)

            # Centre sample and prototype by their own feature means
            x_dev  = x          - x.mean(1, keepdim=True)          # (N, F)
            mu_dev = proto_flat - proto_flat.mean(1, keepdim=True)  # (K*m, F)

            x_a  = x_dev[:, pa]    # (N, P)
            x_b  = x_dev[:, pb]
            mu_a = mu_dev[:, pa]   # (K*m, P)
            mu_b = mu_dev[:, pb]

            cross = (x_a.unsqueeze(1) * mu_b.unsqueeze(0)
                   - x_b.unsqueeze(1) * mu_a.unsqueeze(0))  # (N, K*m, P)

            return torch.sqrt((w_pairs * cross ** 2).sum(-1) + eps)  # (N, K*m)

        def forward(self, x):
            """
            Multi-prototype AVM scoring.

            For each class k with m prototypes {μ_k1,...,μ_km}:
              score_k(x) = Σ_j  τ_k / (d(x, μ_kj) + τ_k·ε)
                         = Σ_j  1 / (d(x, μ_kj)/τ_k + ε)

            This is the sum of inverse-distances over all prototypes of
            class k — a class is close if ANY of its prototypes is close.
            When m=1 reduces exactly to the original model.

            Returns (proba, feat_weights, embeddings).
            """
            tau  = torch.exp(self.log_tau)      # scalar or (K,)
            wb   = self._branch_weights()
            wf   = self._feat_weights()
            N    = x.shape[0]
            K, m = self.K, self.m

            # Combined features for query batch
            fx = self._combined_features(x, wb, wf)           # (N, D)
            D  = fx.shape[1]

            # Flatten prototypes (K, m, F) → (K*m, F) for combined features
            proto_flat = self.centroids.reshape(K * m, self.n_feat)
            fy_flat    = self._combined_features(proto_flat, wb, wf)  # (K*m, D)

            # Pairwise distances (N, K*m) via matmul identity
            sq_x = (fx ** 2).sum(1, keepdim=True)              # (N, 1)
            sq_y = (fy_flat ** 2).sum(1)                       # (K*m,)
            dot  = fx @ fy_flat.T                              # (N, K*m)
            d_flat = torch.sqrt(torch.clamp(
                sq_x + sq_y[None, :] - 2.0 * dot, min=self.eps))  # (N, K*m)

            # Reshape to (N, K, m) and aggregate per class
            d = d_flat.reshape(N, K, m)                        # (N, K, m)

            if K > 1 and tau.dim() > 0:
                # Per-class tau: (K,) → (1, K, 1) for broadcasting
                tau_e = tau.reshape(1, K, 1)
            else:
                tau_e = tau

            raw = (1.0 / (d / tau_e + self.eps)).sum(2)

            if self.use_triangle:
                tau_tri = torch.exp(self.log_tau_tri)
                t_flat  = self._triangle_score(x, proto_flat)
                t       = t_flat.reshape(N, K, m)
                tri_raw = (1.0 / (t / tau_tri + self.eps)).sum(2)
                raw_n   = raw     / raw.sum(-1, keepdim=True).clamp(min=self.eps)
                tri_n   = tri_raw / tri_raw.sum(-1, keepdim=True).clamp(min=self.eps)
                tw      = getattr(self, '_tri_weight', 0.5)
                raw     = (1.0 - tw) * raw_n + tw * tri_n

            proba = raw / raw.sum(-1, keepdim=True)
            return proba, wf, fx

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

        For each anchor i, pulls all same-class embeddings closer and pushes
        all different-class embeddings further in the combined vector space.

        L = sum_i { -1/|P(i)| * sum_{p in P(i)} log [
              exp(sim(z_i, z_p) / T) /
              sum_{a != i} exp(sim(z_i, z_a) / T)
            ]}

        Where:
          sim = cosine similarity between L2-normalised embeddings
          P(i) = set of indices with same label as i (excluding i itself)
          T = temperature (lower = sharper, more discriminative)

        Why this helps on imbalanced data:
          NLL only penalises wrong centroid predictions.
          SupCon explicitly pulls the ~10 minority class samples in a batch
          together while pushing them away from all ~246 majority samples.
          The minority centroid becomes a tight, stable cluster rather than
          a diffuse cloud.
        """
        # L2-normalise embeddings for cosine similarity
        z = nn.functional.normalize(embeddings, dim=1)  # (N, D)
        N = z.shape[0]

        # Cosine similarity matrix scaled by temperature
        sim = (z @ z.T) / self.supcon_temp             # (N, N)

        # Build same-class mask (positive pairs), exclude diagonal
        labels    = targets.unsqueeze(1)               # (N, 1)
        pos_mask  = (labels == labels.T).float()       # (N, N)
        pos_mask.fill_diagonal_(0)                     # exclude self

        # Number of positives per anchor — skip anchors with no positives
        n_pos = pos_mask.sum(1)                        # (N,)
        valid = n_pos > 0

        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device)

        # Subtract max for numerical stability (log-sum-exp trick)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        exp_sim    = torch.exp(sim - sim_max.detach())

        # Zero out self-similarity from denominator
        eye        = torch.eye(N, device=embeddings.device)
        exp_sim    = exp_sim * (1.0 - eye)

        # Log probability of each positive pair
        log_prob   = (sim - sim_max.detach()) - torch.log(
            exp_sim.sum(1, keepdim=True).clamp(min=1e-10))

        # Mean log-prob over positives per anchor
        mean_log_prob = (pos_mask * log_prob).sum(1) / n_pos.clamp(min=1)

        return -mean_log_prob[valid].mean()

    def _loss(self, probs, targets, class_weights, fw=None, embeddings=None):
        """
        fw: pre-computed feat_weights tensor from the forward pass.
        Passing it avoids a third softmax call per batch — forward() already
        computed it, _loss() needs it for fw_reg. If None, recomputes.
        """
        K   = probs.shape[1]
        nll = nn.NLLLoss(weight=class_weights)(
                  torch.log(probs.clamp(1e-10)), targets)
        s   = self.label_smoothing / K
        ce  = (1 - self.label_smoothing + s) * nll + s * (
                   -torch.log(probs.clamp(1e-10)).mean())

        # Feature weight regularisation
        _fw = fw if fw is not None else self.net_._feat_weights()
        fw_reg = self.feat_weight_reg * (_fw - 1.0).pow(2).mean()

        # Centroid anchor
        c_anchor = self.centroid_reg * (
            self.net_.centroids - self.net_.centroid_anchor
        ).pow(2).mean()

        # Centroid separation — cached per epoch, reused across batches.
        # Centroids change by tiny amounts per batch so the approximation
        # error is negligible vs the cost of recomputing n_batch times.
        if self.net_._sep_cache is None:
            if self.net_.K > 1:
                # Inter-class separation — push all prototypes of different
                # classes apart from each other.
                # Flatten (K, m, F) → (K*m, F) for pairwise cdist.
                K, m = self.net_.K, self.net_.m
                proto_flat = self.net_.centroids.reshape(K * m, self.net_.n_feat)
                c_dists    = torch.cdist(proto_flat, proto_flat, p=2)

                # Build mask: pairs from different classes
                class_idx   = torch.arange(K, device=c_dists.device).repeat_interleave(m)
                diff_class  = (class_idx.unsqueeze(0) != class_idx.unsqueeze(1)).float()
                upper       = torch.triu(torch.ones_like(diff_class), diagonal=1)
                inter_mask  = diff_class * upper

                if inter_mask.sum() > 0:
                    inter = -self.centroid_sep * (
                        c_dists * inter_mask).sum() / inter_mask.sum()
                else:
                    inter = torch.tensor(0.0)

                # Intra-class separation — push prototypes within same class
                # apart so they spread to cover different sub-clusters.
                # Only meaningful when m > 1.
                intra = torch.tensor(0.0)
                if m > 1 and self.intra_sep > 0.0:
                    for k in range(K):
                        proto_k = self.net_.centroids[k]   # (m, F)
                        d_intra = torch.cdist(proto_k, proto_k, p=2)
                        mask_k  = torch.triu(
                            torch.ones(m, m, device=d_intra.device), diagonal=1)
                        if mask_k.sum() > 0:
                            intra = intra - self.intra_sep * (
                                d_intra * mask_k).sum() / mask_k.sum()

                self.net_._sep_cache = (inter + intra).detach()
            else:
                self.net_._sep_cache = torch.tensor(0.0)
        c_sep = self.net_._sep_cache

        if self.ordinal and K > 2:
            emd = self._ordinal_loss(probs, targets)
            ce  = (1.0 - self.ordinal_weight) * ce + self.ordinal_weight * emd

        if self.supcon and embeddings is not None:
            sc  = self._supcon_loss(embeddings, targets)
            ce  = (1.0 - self.supcon_weight) * ce + self.supcon_weight * sc

        # Orthogonalization — Gram matrix penalty on branch outputs.
        # Splits the combined vector into 6 branch chunks and penalises
        # pairwise correlation between branches. Encourages each branch
        # to encode a unique geometric dimension rather than redundantly
        # encoding the same magnitude/extremeness signal.
        #
        # L_ortho = ||G - I||²_F  where G_ij = normalised dot product
        #           between branch i and branch j outputs (cosine similarity).
        # Off-diagonal G_ij → 0 means branches are decorrelated.
        # Diagonal G_ii = 1 always (self-similarity).
        if self.ortho_reg > 0.0 and embeddings is not None:
            n_branches = 6
            F_branch   = embeddings.shape[1] // n_branches
            # Split into (N, n_branches, F_branch), normalise per-sample
            # along F_branch dim — gives per-sample branch direction vectors.
            # Average Gram matrix across batch for stable decorrelation signal.
            chunks  = embeddings[:, :n_branches * F_branch].reshape(
                -1, n_branches, F_branch)                # (N, B, F)
            normed  = nn.functional.normalize(chunks, dim=2)  # (N, B, F)
            # Batch-averaged Gram: (N, B, B) → mean over N → (B, B)
            G       = torch.bmm(normed, normed.transpose(1, 2)).mean(0)
            I       = torch.eye(n_branches, device=G.device)
            ortho   = self.ortho_reg * (G - I).pow(2).sum()

            return ce + fw_reg + c_anchor + c_sep + ortho

        return ce + fw_reg + c_anchor + c_sep

    # ── composite validation metric ───────────────────────────────────────────

    def _val_score(self, y_true, y_pred):
        """
        Composite validation metric computed from a single confusion matrix.

        sklearn's accuracy_score, f1_score(macro), f1_score(weighted) each
        build their own internal representation — three passes over the data.
        Computing the confusion matrix once and deriving all three metrics from
        it is 4.5x faster on typical validation set sizes (150-500 samples).
        """
        from sklearn.metrics import confusion_matrix
        labels = np.arange(len(self.classes_))
        cm     = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
        tp     = np.diag(cm)
        total  = cm.sum()
        sup    = cm.sum(1)          # per-class support
        pred_s = cm.sum(0)          # per-class predicted count

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

        self.classes_ = np.unique(y)
        K      = len(self.classes_)
        n_feat = X.shape[1]

        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([label_to_idx[c] for c in y], dtype=np.int64)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Stratified train/val split — minimum K*5 samples per class in val
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
                dtype=torch.long if long else torch.float32)

        cw   = compute_class_weight('balanced', classes=self.classes_,
                                    y=y[tr_idx])
        cw_t = torch.tensor(cw, dtype=torch.float32).clamp(
            max=self.weight_cap)
        cw_t = cw_t / cw_t.sum() * K

        m = max(1, int(self.n_prototypes))

        # Initialise prototypes — shape (K, m, F).
        # Each prototype starts at the class mean. When m>1, prototypes
        # are identical at init; intra-class separation pushes them apart
        # during training, guided by the NLL gradient.
        # Small jitter breaks symmetry so they don't stay identical.
        rng_proto = np.random.default_rng(
            (self.random_state or 0) + 1)
        class_means = np.stack([
            X_tr_n[y_tr == i].mean(0) for i in range(K)])  # (K, F)
        proto_init  = np.tile(
            class_means[:, None, :], (1, m, 1))             # (K, m, F)
        if m > 1:
            jitter = rng_proto.normal(
                0, 0.05, proto_init.shape).astype(np.float32)
            proto_init = proto_init + jitter

        centroids = tt(proto_init)    # (K, m, F)

        self.net_ = self._Net(centroids, K, n_feat, m=m,
                              per_class_tau=self.per_class_tau,
                              use_triangle=self.use_triangle)
        self.net_._tri_weight = float(self.tri_weight)

        opt = optim.Adam(self.net_.parameters(),
                         lr=self.lr, weight_decay=self.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                  opt, T_0=30, T_mult=2, eta_min=1e-5)

        # Pre-bake tensors — no DataLoader overhead
        X_tr_t = tt(X_tr_n)
        y_tr_t = tt(y_tr, long=True)
        X_va_t = tt(X_va_n)
        # Cache val combined features — X_va_n never changes so
        # _make_combined(X_va_n) produces the same result every val check.
        # Pre-compute once here; reuse in every validation epoch.
        # NOTE: this cache is in PyTorch space (for net forward), not numpy.
        # It is invalidated implicitly — we never call _make_combined on val
        # during training, only net_(X_va_t) which uses learnable weights.
        # The PyTorch path doesn't use _make_combined so this is fine.
        N_tr   = len(X_tr_t)
        bs     = self.batch_size
        n_batch = max(1, N_tr // bs)

        # val_every: validate every N epochs to reduce overhead.
        # patience counts EPOCHS not validation events — divide internally
        # so patience=60 always means 60 epochs regardless of val_every.
        val_every   = 5
        best_f1, best_state, wait = -1.0, None, 0
        WARMUP = 20

        for epoch in range(self.epochs):
            perm   = rng.permutation(N_tr)
            X_shuf = X_tr_t[perm]
            y_shuf = y_tr_t[perm]

            self.net_.train()
            self.net_._sep_cache = None   # invalidate once per epoch
            for i in range(n_batch):
                xb = X_shuf[i*bs:(i+1)*bs]
                yb = y_shuf[i*bs:(i+1)*bs]
                opt.zero_grad()
                proba, fw, emb = self.net_(xb)
                self._loss(proba, yb, cw_t, fw=fw,
                           embeddings=emb if self.supcon else None).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.net_.parameters(), 1.0)
                opt.step()
            sch.step()

            # EMA centroid update — after each epoch, blend learnable centroid
            # positions toward the actual mean of embedded training samples.
            # This stabilises minority class centroids whose backprop gradient
            # signal is weak (few samples → small contribution to NLL loss).
            # EMA provides a direct geometric pull toward the empirical cluster.
            if self.ema_centroids:
                self.net_.eval()
                with torch.no_grad():
                    for cls in range(K):
                        mask     = (y_tr_t == cls)
                        if mask.sum() == 0:
                            continue
                        cls_X    = X_tr_t[mask]          # (n_k, F)
                        protos   = self.net_.centroids[cls]  # (m, F)

                        if m == 1:
                            # Single prototype — direct EMA
                            cls_mean = cls_X.mean(0)
                            self.net_.centroids.data[cls, 0] = (
                                self.ema_beta       * protos[0]
                                + (1.0 - self.ema_beta) * cls_mean)
                        else:
                            # Multiple prototypes — soft assignment.
                            # Each sample weighted by proximity to each proto.
                            # d: (n_k, m) distances in raw normalised space
                            d_proto = torch.cdist(cls_X, protos, p=2)  # (n_k, m)
                            resp    = torch.softmax(-d_proto, dim=1)    # (n_k, m)
                            for j in range(m):
                                w    = resp[:, j:j+1]                   # (n_k, 1)
                                w_sum = w.sum().clamp(min=1e-10)
                                wm   = (w * cls_X).sum(0) / w_sum       # (F,)
                                self.net_.centroids.data[cls, j] = (
                                    self.ema_beta       * protos[j]
                                    + (1.0 - self.ema_beta) * wm)

            # Skip validation on non-val epochs — still count toward patience
            if (epoch + 1) % val_every != 0 and epoch < self.epochs - 1:
                # Patience counts epochs, not val events
                if epoch >= WARMUP and best_state is not None:
                    wait += 1
                if self.verbose and (epoch + 1) % 10 == 0:
                    bw = self.net_._branch_weights().detach().numpy()
                    print(f"  epoch {epoch+1:3d}  (val skipped)"
                          f"  branches=[lin:{bw[0]:.2f} cir:{bw[1]:.2f}"
                          f" bnd:{bw[2]:.2f} shp:{bw[3]:.2f} sq:{bw[4]:.2f} tac:{bw[5]:.2f}]")
                if wait >= self.patience:
                    if self.verbose:
                        print(f"  early stopping at epoch {epoch+1}")
                    break
                continue

            self.net_.eval()
            with torch.no_grad():
                vp, _, _ = self.net_(X_va_t)
            vf1 = self._val_score(y_va, vp.numpy().argmax(1))

            if vf1 > best_f1:
                best_f1    = vf1
                best_state = {k: v.clone() for k, v
                              in self.net_.state_dict().items()}
                wait = 0
            else:
                if epoch >= WARMUP:
                    wait += 5   # account for skipped epochs

            if self.verbose and (epoch + 1) % 10 == 0:
                bw = self.net_._branch_weights().detach().numpy()
                print(f"  epoch {epoch+1:3d}  val_f1={vf1:.4f}"
                      f"  branches=[lin:{bw[0]:.2f} cir:{bw[1]:.2f}"
                      f" bnd:{bw[2]:.2f} shp:{bw[3]:.2f} sq:{bw[4]:.2f} tac:{bw[5]:.2f}]")

            if wait >= self.patience:
                if self.verbose:
                    print(f"  early stopping at epoch {epoch+1}")
                break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.best_val_f1_ = best_f1
        # Store which branches/features were pruned for inspection

        # Extract learned parameters
        with torch.no_grad():
            bw = self.net_._branch_weights().numpy().copy()
            fw = self.net_._feat_weights().numpy().copy()
            self.branch_weights_ = bw
            self.feat_weights_   = fw
            self.tau_            = torch.exp(self.net_.log_tau).numpy().copy()

        # Build numpy combined vector — IDENTICAL structure to _combined_features
        # so training and inference use exactly the same metric.
        _bw  = bw.copy()
        _fw  = fw.copy()
        _bw  = bw.copy()
        _fw  = fw.copy()
        s_lin = np.float32(np.sqrt(_bw[0]))   # linear
        s_cir = np.float32(np.sqrt(_bw[1]))   # circular
        s_b   = np.float32(np.sqrt(_bw[2]))   # boundary
        s_s   = np.float32(np.sqrt(_bw[3]))   # shape
        s_sq  = np.float32(np.sqrt(_bw[4]))   # quadratic
        s_tac = np.float32(np.sqrt(_bw[5]))   # tanh_arccos
        sqfw  = np.sqrt(_fw).astype(np.float32)

        def _make_combined(xn, batch_size=10_000):
            """
            Numpy equivalent of _Net._combined_features — 6 branches.
            [lin, cir, boundary, shape, quadratic, tanh_arccos]
            Identical to PyTorch — no train/inference mismatch.
            """
            def _chunk(chunk):
                eps  = 1e-7
                lin  = chunk * sqfw
                cir  = np.sqrt(np.clip(1.0 - chunk*chunk, eps, None)) * sqfw
                sqd  = chunk * chunk * sqfw
                norm_sq = (chunk ** 2).sum(1, keepdims=True)
                bound   = np.sqrt(
                    (norm_sq - 2.0 * chunk + 1.0).clip(min=eps))
                mu  = chunk.mean(1, keepdims=True)
                sd  = chunk.std(1,  keepdims=True).clip(min=eps)
                shp = (chunk - mu) / sd
                a   = np.arccos(np.clip(chunk, -1+eps, 1-eps))
                tac = np.tanh(a * 0.8) * sqfw
                return np.concatenate([
                    s_lin*lin, s_cir*cir, s_b*bound,
                    s_s*shp,   s_sq*sqd,  s_tac*tac
                ], axis=1).astype(np.float32)

            if len(xn) <= batch_size:
                return _chunk(xn)
            return np.vstack([
                _chunk(xn[i:i+batch_size])
                for i in range(0, len(xn), batch_size)])

        self._make_combined = _make_combined

        # Extract learned prototype positions in combined space
        # centroids: (K, m, F) → flatten to (K*m, F) for _make_combined
        raw_c_n     = self.net_.centroids.detach().numpy()     # (K, m, F)
        init_c_n    = self.net_.centroid_anchor.numpy()        # (K, m, F)
        raw_c_flat  = raw_c_n.reshape(K * m, n_feat)
        self.centroids_combined_ = _make_combined(
            raw_c_flat).astype(np.float32)                     # (K*m, D)
        self.n_prototypes_ = m   # actual m used (post-fit)

        # Drift: mean distance each prototype moved from its anchor
        drift_flat = np.sqrt(
            ((raw_c_n - init_c_n) ** 2).sum(-1))              # (K, m)
        self.centroid_drift_ = drift_flat.mean(1)              # (K,) mean over protos

        counts_tr = np.array([(y_tr == i).sum() for i in range(K)],
                              dtype=np.float32)
        if self.verbose:
            for i in range(K):
                proto_drifts = "  ".join(
                    f"p{j}={drift_flat[i,j]:.4f}" for j in range(m))
                print(f"  centroid cls{self.classes_[i]}: "
                      f"n={int(counts_tr[i])}  drift=[{proto_drifts}]")

        # Regularised full Mahalanobis covariance in combined space
        #
        # Diagonal Mahalanobis rescales each dimension independently —
        # equivalent to an axis-aligned ellipsoid. It cannot represent
        # correlated features (e.g. high-quality wine = high alcohol AND
        # low volatile acidity jointly). Full covariance captures rotation:
        # the decision boundary can align with the actual cluster orientation.
        #
        # Estimation strategy: Regularised Discriminant Analysis (RDA)
        #   Σ_k = (1-α)*S_k + α*S_pooled
        #
        #   α → 0: full per-class QDA (may be ill-conditioned for small n_k)
        #   α → 1: shared pooled covariance (LDA)
        #   α = 0.5: balanced blend (default)
        #
        # Then ridge regularise the blend:
        #   Σ_k_reg = (1-r)*Σ_k + r*I * mean(diag(Σ_k))
        #
        # This ensures positive-definiteness even when n_k < dim.
        # Cholesky decomposition used for numerically stable inversion.
        # Build training combined vectors once — reused for both
        # Mahalanobis covariance estimation and FAISS index construction.
        X_tr_c_post = _make_combined(X_tr_n)   # (N_tr, D)

        if self.mahalanobis:
            X_tr_c = X_tr_c_post
            D      = X_tr_c.shape[1]

            # Pooled covariance — weighted sum of per-class covariances
            S_pooled = np.zeros((D, D), dtype=np.float64)
            for i in range(K):
                cls_pts = X_tr_c[y_tr == i].astype(np.float64)
                if len(cls_pts) < 2:
                    continue
                c = cls_pts - cls_pts.mean(0, keepdims=True)
                S_pooled += (c.T @ c)
            S_pooled /= (len(X_tr_c) - K)   # unbiased pooled estimate

            self.cov_inv_  = np.zeros((K, D, D), dtype=np.float32)
            alpha          = float(self.mahal_alpha)
            ridge          = float(self.mahal_reg)

            for i in range(K):
                cls_pts = X_tr_c[y_tr == i].astype(np.float64)
                n_k     = len(cls_pts)

                if n_k < 2:
                    # Only 1 sample — use pooled covariance entirely
                    S_k = S_pooled.copy()
                else:
                    c   = cls_pts - cls_pts.mean(0, keepdims=True)
                    S_k = (c.T @ c) / (n_k - 1)

                # RDA blend: mix per-class and pooled
                S_blend = (1.0 - alpha) * S_k + alpha * S_pooled

                # Ridge regularisation toward scaled identity
                # Prevents ill-conditioning when n_k < D
                scale          = np.diag(S_blend).mean().clip(min=1e-10)
                S_reg          = ((1.0 - ridge) * S_blend
                                  + ridge * scale * np.eye(D))

                # Cholesky inversion — numerically stable for PD matrices
                try:
                    L     = np.linalg.cholesky(S_reg)
                    L_inv = np.linalg.solve(L, np.eye(D))
                    self.cov_inv_[i] = (L_inv.T @ L_inv).astype(np.float32)
                except np.linalg.LinAlgError:
                    # Fallback: pseudo-inverse if Cholesky fails
                    self.cov_inv_[i] = np.linalg.pinv(
                        S_reg).astype(np.float32)

            if self.verbose:
                print(f"  Mahalanobis: full RDA "
                      f"(alpha={alpha:.2f} ridge={ridge:.0e} D={D})")
        else:
            self.cov_inv_ = None

        # FAISS index — reuses X_tr_c_post computed above
        self.lambda_ = self.lam_floor
        if not _HAS_FAISS:
            if self.verbose:
                print("  faiss not installed — KNN disabled, lambda=1.0")
            self.lambda_ = 1.0
            self.index_  = None
        else:
            X_tr_c  = X_tr_c_post
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

            # Lambda selection — three modes
            if self.entropy_lambda:
                # AVM confidence gate — no training needed.
                # lambda(x) = lam_confident - (lam_confident - lam_uncertain) * H(x)
                # where H(x) = normalised entropy of AVM probability vector.
                # High AVM entropy (uncertain) → low lambda (trust KNN more).
                # Low AVM entropy (confident)  → high lambda (trust AVM more).
                self.lambda_ = None   # signals entropy gate in predict_proba
                if self.verbose:
                    print(f"  lambda mode=entropy  "
                          f"confident={self.lam_confident}  "
                          f"uncertain={self.lam_uncertain}")

            else:
                # Scalar lambda search (original behaviour)
                X_va_c  = _make_combined(X_va_n)
                avm_val = self._avm_proba(X_va_c)
                knn_val = self._knn_proba(X_va_c)
                best_lam_f1, best_lam = -1.0, self.lam_floor
                step = 0.05
                for lam_c in np.arange(
                        self.lam_floor, 1.0 + step / 2, step):
                    lam_c = float(np.round(lam_c, 6))
                    p  = lam_c * avm_val + (1.0 - lam_c) * knn_val
                    f1 = self._val_score(y_va, p.argmax(1))
                    if f1 > best_lam_f1:
                        best_lam_f1 = f1
                        best_lam    = lam_c
                self.lambda_ = best_lam
                if self.verbose:
                    print(f"  lambda search (floor={self.lam_floor}): "
                          f"best={best_lam:.2f}"
                          f"  val_f1={best_lam_f1:.4f}")

        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def _avm_proba(self, combined):
        """
        Multi-prototype AVM scoring.
        centroids_combined_ shape: (K*m, D)
        Aggregates inverse-distances per class by summing over prototypes.
        """
        K   = len(self.classes_)
        m   = self.n_prototypes_
        N   = combined.shape[0]

        if self.cov_inv_ is not None:
            # Full Mahalanobis — one cov_inv per class applied to each proto.
            # Use float64 intermediates — float32 overflows on high-dimensional
            # combined vectors (digits: 384-dim × large covariance values).
            d_flat = np.zeros((N, K * m), dtype=np.float64)
            comb64 = combined.astype(np.float64)
            for km in range(K * m):
                k    = km // m
                diff = comb64 - self.centroids_combined_[km].astype(np.float64)
                tmp  = diff @ self.cov_inv_[k].astype(np.float64)
                d_flat[:, km] = np.sqrt(
                    np.maximum((tmp * diff).sum(1), 0.0))
            d_flat = d_flat.astype(np.float32)
        else:
            a_sq   = (combined ** 2).sum(1, keepdims=True)
            b_sq   = (self.centroids_combined_ ** 2).sum(1)
            ab     = combined @ self.centroids_combined_.T
            d_flat = np.sqrt(np.maximum(
                a_sq + b_sq[None, :] - 2.0 * ab, 0.0)).astype(np.float32)
            # d_flat: (N, K*m)

        # Aggregate: sum 1/(d+eps) per class over its m prototypes
        d_km  = d_flat.reshape(N, K, m)              # (N, K, m)
        raw   = (1.0 / (d_km + 1e-10)).sum(2)        # (N, K)
        return raw / raw.sum(1, keepdims=True)

    def _knn_proba(self, combined, search_batch=50_000):
        if self.index_ is None:
            raise RuntimeError("FAISS index not built.")
        n_test = combined.shape[0]
        K      = len(self.classes_)
        proba  = np.zeros((n_test, K), dtype=np.float32)
        for start in range(0, n_test, search_batch):
            end   = min(start + search_batch, n_test)
            batch = combined[start:end]
            dist, idx = self.index_.search(batch, self.k)
            w = 1.0 / (dist + 1e-10)
            w = w / w.sum(1, keepdims=True)
            t_idx = np.repeat(np.arange(end - start), self.k)
            np.add.at(proba[start:end],
                      (t_idx, self.train_class_idx_[idx.ravel()]),
                      w.ravel())
        return proba

    def predict_proba(self, X):
        check_is_fitted(self, 'centroids_combined_')
        X        = np.asarray(X, dtype=np.float32)
        combined = self._make_combined(self._norm_apply(X))
        avm      = self._avm_proba(combined)

        if self.lambda_ is not None and (
                self.lambda_ >= 1.0 or self.index_ is None):
            return avm

        if self.index_ is None:
            return avm

        knn = self._knn_proba(combined)

        if self.entropy_lambda:
            # AVM entropy gate — per-sample lambda based on AVM confidence
            K   = len(self.classes_)
            # Normalised Shannon entropy: 0=confident, 1=maximally uncertain
            H   = -(avm * np.log(avm.clip(1e-10))).sum(1) / np.log(K)
            lam = (self.lam_confident
                   - (self.lam_confident - self.lam_uncertain) * H)
            lam = lam.clip(self.lam_uncertain, self.lam_confident)
            return lam[:, None] * avm + (1 - lam[:, None]) * knn

        else:
            # Scalar lambda (default)
            lam = self.lambda_ if self.lambda_ is not None else self.lam_floor
            return lam * avm + (1.0 - lam) * knn

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def summary(self, feature_names=None):
        check_is_fitted(self, 'branch_weights_')
        names = (list(feature_names) if feature_names is not None
                 else [f"f{i}" for i in range(len(self.feat_weights_))])
        bw  = self.branch_weights_
        lam = getattr(self, 'lambda_', self.lam_floor)
        tau = self.tau_
        tau_str = (f"{float(tau):.3f}" if np.ndim(tau) == 0 or np.size(tau) == 1
                   else "[" + " ".join(
                       f"cls{i}:{v:.3f}" for i, v in enumerate(tau)) + "]")
        lam_str = (f"{lam:.3f}" if lam is not None
                   else "[entropy]")
        lines = [
            f"  tau={tau_str}  lambda={lam_str}"
            f"  entropy_lambda={self.entropy_lambda}",
            f"  ordinal={self.ordinal}"
            f"  per_class_tau={self.per_class_tau}"
            f"  mahalanobis={'full' if self.mahalanobis else 'off'}"
            f"  alpha={self.mahal_alpha}"
            f"  ortho_reg={self.ortho_reg}",
            f"  supcon={self.supcon}"
            f"  ema_centroids={self.ema_centroids}"
            f"  n_prototypes={getattr(self, 'n_prototypes_', self.n_prototypes)}"
            f"  triangle={self.use_triangle}",
            f"  Branch weights: lin={bw[0]:.3f} cir={bw[1]:.3f} bnd={bw[2]:.3f}"
            f" shp={bw[3]:.3f} sq={bw[4]:.3f} tac={bw[5]:.3f}",
        ]
        order = np.argsort(self.feat_weights_)[::-1]
        lines.append("  Top feature weights:")
        for i in order[:5]:
            lines.append(
                f"    {names[i]:<26} {self.feat_weights_[i]:.4f}")
        if hasattr(self, 'centroid_drift_') and self.centroid_drift_ is not None:
            d_str = "  ".join(
                f"class{c}={v:.4f}" for c, v in
                zip(self.classes_, self.centroid_drift_))
            lines.append(f"  Centroid drift: {d_str}")
        if hasattr(self, 'branch_frozen_') and self.branch_frozen_.any():
            bnames = ['lin', 'cir', 'boundary', 'shape']
            pruned = [bnames[i] for i in range(6) if self.branch_frozen_[i]]
            lines.append(f"  Pruned branches: {pruned}")
        if hasattr(self, 'feat_frozen_') and self.feat_frozen_.any():
            n_pruned = self.feat_frozen_.sum()
            lines.append(f"  Pruned features: {n_pruned}/{len(self.feat_weights_)}")
        return "\n".join(lines)
