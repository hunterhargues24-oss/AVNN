"""
LearningAVNN
============
Geometric tabular classifier combining a learned Angular Vector Machine (AVM)
prototype space with FAISS-accelerated KNN and optional Fisher/geodesic heads.

Two-stage design (note the asymmetry — it is deliberate)
-------------------------------------------------------
  TRAIN     prototype positions, branch gates, feature weights and per-class
            tau are learned under Euclidean inverse-distance scoring in the
            AVM feature space (cheap, well-conditioned, no covariance needed).
  INFERENCE the frozen prototypes are re-scored with a full QDA posterior:
            per-prototype RDA covariance estimated post-fit, then

              log p(k|x) ∝ log π_k
                         + logsumexp_{p∈k}[ −0.5·log|Σ_p| − 0.5·d²_M(x, c_p) ]

            (the −0.5·D·log 2π and −log m constants cancel in the softmax over
            k; the logsumexp marginalises the m prototypes of a class as an
            equal-weight Gaussian mixture).
  Prototypes are therefore Euclidean-optimal but scored under QDA. Closing this
  loop (mid-training covariance re-estimation) is possible but unimplemented.
  tau is a TRAINING-only sharpness knob — inference has never used it.

Architecture
------------
  Input → MinMax normalise [-1,1]
       → AVM space:  three-branch vector (tac + boundary + circular)
            → per-prototype QDA posterior over K×m learnable prototypes
       → KNN space:  separate frozen three-branch vector (linear+shape+quad,
                     plus optional log/rank/clr), FAISS inverse-distance vote
       → fusion: avm_only | λ-blend (avm+knn) | confidence-gate | static
       → optional prior tempering (prior_temp) on the fused posterior

AVM Branches  (global structure — centroid proximity)
-----------------------------------------------------
  tanh_arccos  tanh(arccos(x)·0.8)·√fw   monotone nonlinear, centre-amplified
               high at x≈-1, zero at x≈+1; most sensitive near x=0
  boundary     √(‖x‖²-2x+1)              cross-feature coupling via ‖x‖²
               optional dual_boundary adds negative-face distances (2F).
               NOT feature-weighted — the ‖x‖² coupling makes per-feature
               weighting ill-defined (intentional asymmetry; consistent across
               the torch train path and the numpy inference path).
  circular     √(1-x²)·√fw               symmetric extremeness at ±1

  Gates: independent normalised-sigmoid (not softmax) — tac and boundary
  both initialised with +0.5 bias head start over circular.

KNN Branches  (local structure — neighbourhood similarity, frozen)
------------------------------------------------------------------
  linear       x·√fw           signed magnitude, direct local proximity
  shape        (x-μ)/σ         within-sample relative profile fingerprint
  quadratic    x²·√fw          nonlinear extremeness, different from circular
  optional:    log (multiplicative) · rank (monotone) · clr (compositional) ·
               interaction (cross-feature products x_i·x_j, i<j, random-
               projected to interaction_dim coords and z-scored — the cheap
               probe for whether nonlinear feature interactions carry signal)

  KNN branch weights are parameters but receive no gradient (the KNN space is
  not used in the training forward pass). They encode a fixed geometric view.

Learnable Prototypes
--------------------
  K×m prototype positions (nn.Parameter). n_prototypes=1 → single centroid.
  n_prototypes>1 → multimodal class representation (e.g. bimodal quality).
  centroid_sep / intra_sep — recomputed (ATTACHED) every batch, prototypes
  pushed apart during training.

  boundary_init (opt-in, m>1) — prototype 0 at the class centroid (interior
  anchor), prototypes 1..m-1 at the lowest-positive-margin training points
  (boundary medoids).

Discriminant / covariance
-------------------------
  Per-prototype RDA:  Σ_p = (1−α)·S_p + α·S_pooled, ridge-regularised, then
  Cholesky-inverted; log|Σ_p| is captured from the Cholesky factor and feeds
  the QDA volume term, scaled by logdet_weight (1.0 = full QDA; 0.0 drops the
  volume term for robustness when D is large relative to per-prototype n, where
  the determinant is dominated by ridge-floored noise directions). Covariance
  is estimated on per-column-STANDARDISED combined features (avm_scale_) so the
  ridge / RDA pooling is not dominated by the boundary block (∝ √F scale). The
  same standardisation is applied to the query and the centroids inside
  _avm_proba. Σ_p is centred on the soft-assigned empirical mean while the
  Gaussian component is centred on the learned prototype c_p; for m=1 these
  coincide, for m>1 they differ by design.

Imbalance Handling
------------------
  SupCon loss     — pulls same-class AVM embeddings together per batch
  Per-class tau   — per-class training sharpness
  Mahalanobis QDA — full per-class/per-prototype covariance at inference
  Class weights   — balanced NLL capped at weight_cap
  Ordinal EMD     — Earth Mover's Distance for ordered class structures
  EMA centroids   — exponential moving average stabilises minority prototypes
                    (optimizer state for the centroid param is reset after each
                    EMA write so stale Adam momentum can't fight the EMA pull)
  log π_k prior   — built into the QDA score, scaled by prior_weight (set 0 to
                    drop it and lift minority recall on imbalanced data); the
                    separate prior_temp still tempers the post-fusion decision
  Score form      — avm_score='inverse_distance' restores the flat heavy-tailed
                    vote the entropy gates were calibrated to (vs sharp QDA)
  val_macro_bias  — early stopping biased toward macro F1 over accuracy

Known asymmetries / caveats
---------------------------
  • Train (Euclidean) vs inference (QDA) metric mismatch — see Two-stage design.
  • ortho_reg groups embeddings by the THREE semantic AVM branches; under dual
    boundary the 2F block is collapsed to its positive faces before the Gram.
  • Label smoothing is unweighted while NLL is class-weighted (left deliberate).
  • FAISS IVF k-means is not seeded by random_state (KNN head non-deterministic).

Key Parameters
--------------
  k              int    KNN neighbours
  n_prototypes   int    prototypes per class
  boundary_init  bool   boundary-medoid placement for m>1 (default False)
  mahal_alpha    float  RDA blend (0=QDA, 1=LDA)
  logdet_weight  float  scales the QDA volume term −0.5·log|Σ_k| (1=full QDA,
                        0=distance+prior only; lower for high-D / low-n)
  avm_score      str    'qda' (sharp softmax, correct) | 'inverse_distance'
                        (flat Σ 1/d_M vote; recovers pre-QDA calibration)
  prior_weight   float  scales the empirical class prior π_k in both score
                        modes (1=full prior; 0=no prior → higher minority recall)
  interaction_dim int|'auto'  output width of the 'interaction' KNN branch
                        ('auto' = n_features); random-projected pairwise products
  lam_floor      float  minimum AVM weight in lambda grid search
  val_macro_bias float  0=accuracy, 1=macro F1 for early stopping
  supcon         bool   supervised contrastive loss
  ema_centroids  bool   EMA prototype stabilisation
  ortho_reg      float  Gram matrix branch decorrelation
  use_dual_boundary bool extend boundary branch to 2F
  prior_temp     float  tempering exponent on the QDA log prior (0 = single
                        built-in prior; >0 over-weights it post-fusion)
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
                 n_prototypes=1, intra_sep=0.05, boundary_init=False,
                 use_dual_boundary=False,
                 ortho_reg=0.01,
                 mahalanobis=True, mahal_reg=1e-6, mahal_alpha=0.5,
                 logdet_weight=1.0, avm_score='qda', prior_weight=1.0,
                 lam_floor=0.5, entropy_lambda=False,
                 lam_confident=0.90, lam_uncertain=0.40,
                 ordinal=False, ordinal_weight=0.5,
                 per_class_tau=True,
                 prior_temp=0.0,
                 val_macro_bias=0.5,
                 use_ivf=True,
                 supcon=False, supcon_weight=0.3, supcon_temp=0.1,
                 ema_centroids=False, ema_beta=0.9,
                 heads=('avm', 'knn'),
                 knn_branches=('linear', 'shape', 'quadratic'),
                 interaction_dim='auto',
                 geodesic_neighbors=10, geodesic_components=5,
                 geodesic_max_fit=4000,
                 device='auto',
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
        self.boundary_init       = boundary_init
        self.use_dual_boundary   = use_dual_boundary
        self.ortho_reg           = ortho_reg
        self.mahalanobis         = mahalanobis
        self.mahal_reg           = mahal_reg
        self.mahal_alpha         = mahal_alpha
        self.logdet_weight       = logdet_weight
        self.avm_score           = avm_score
        self.prior_weight        = prior_weight
        self.lam_floor           = lam_floor
        self.entropy_lambda      = entropy_lambda
        self.lam_confident       = lam_confident
        self.lam_uncertain       = lam_uncertain
        self.ordinal             = ordinal
        self.ordinal_weight      = ordinal_weight
        self.per_class_tau       = per_class_tau
        self.prior_temp          = prior_temp
        self.val_macro_bias      = val_macro_bias
        self.use_ivf             = use_ivf
        self.supcon              = supcon
        self.supcon_weight       = supcon_weight
        self.supcon_temp         = supcon_temp
        self.ema_centroids       = ema_centroids
        self.ema_beta            = ema_beta
        self.heads               = heads
        self.knn_branches        = knn_branches
        self.interaction_dim     = interaction_dim
        self.geodesic_neighbors  = geodesic_neighbors
        self.geodesic_components = geodesic_components
        self.geodesic_max_fit    = geodesic_max_fit
        self.device              = device
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
                     per_class_tau=False,
                     use_dual_boundary=False, eps=1e-7):
            super().__init__()
            self.K, self.m, self.n_feat, self.eps = K, m, n_feat, eps
            self.per_class_tau    = per_class_tau
            self.use_dual_boundary = use_dual_boundary

            self.centroids = nn.Parameter(centroids.clone())
            self.register_buffer("centroid_anchor", centroids.clone())

            if per_class_tau:
                self.log_tau = nn.Parameter(torch.full((K,), -0.5))
            else:
                self.log_tau = nn.Parameter(torch.tensor(-0.5))

            # ── AVM branches (global structure — centroid proximity) ──────────
            # AVM branches: tac, boundary, circular
            self.raw_w_tac = nn.Parameter(torch.tensor(0.5))   # ↑ head start
            self.raw_w_b   = nn.Parameter(torch.tensor(0.5))   # ↑ head start
            self.raw_w_cir = nn.Parameter(torch.tensor(0.0))

            # ── KNN branches (local structure — neighbourhood similarity) ─────
            self.raw_w_lin  = nn.Parameter(torch.tensor(0.0))
            self.raw_w_s    = nn.Parameter(torch.tensor(0.0))
            self.raw_w_sq   = nn.Parameter(torch.tensor(0.0))

            # Feature weights — shared across AVM and KNN spaces
            self.raw_feat_w = nn.Parameter(torch.zeros(n_feat))

        def _avm_branch_weights(self):
            """AVM branches: tac, boundary, circular — normalised sigmoid."""
            raw   = torch.stack([self.raw_w_tac, self.raw_w_b, self.raw_w_cir])
            gates = torch.sigmoid(raw)
            return gates / (gates.sum() + 1e-10)

        def _knn_branch_weights(self):
            """KNN branches: linear, shape, quadratic — normalised sigmoid."""
            raw   = torch.stack([self.raw_w_lin, self.raw_w_s, self.raw_w_sq])
            gates = torch.sigmoid(raw)
            return gates / (gates.sum() + 1e-10)

        def _branch_weights(self):
            """Combined 6-weight view — AVM then KNN."""
            return torch.cat([self._avm_branch_weights(),
                              self._knn_branch_weights()])

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
            The neg distances give the AVM direct signal about proximity
            to the negative face — complementary to the tac branch.
            """
            norm_sq     = (x**2).sum(dim=1, keepdim=True)
            dist_sq_pos = (norm_sq - 2.0*x + 1.0).clamp(min=eps)
            if not dual:
                return torch.sqrt(dist_sq_pos)
            dist_sq_neg = (norm_sq + 2.0*x + 1.0).clamp(min=eps)
            # Interleave: [pos_f0, neg_f0, pos_f1, neg_f1, ...]
            stacked = torch.stack([dist_sq_pos, dist_sq_neg], dim=2)  # (N,F,2)
            return torch.sqrt(stacked.reshape(stacked.shape[0], -1))   # (N,2F)

        def _avm_features(self, x, wf):
            """
            AVM combined vector — global structure branches.
            tac + boundary (F or 2F if dual) + circular.
            """
            eps  = self.eps
            sqwf = wf.sqrt()
            wa   = self._avm_branch_weights()

            a   = torch.acos(torch.clamp(x, -1+eps, 1-eps))
            tac = torch.tanh(a * 0.8) * sqwf

            bnd = self._boundary_dists(x, eps, dual=self.use_dual_boundary)

            cir = torch.sqrt(torch.clamp(1.0 - x*x, min=eps)) * sqwf

            return torch.cat([
                wa[0].sqrt() * tac,
                wa[1].sqrt() * bnd,
                wa[2].sqrt() * cir,
            ], dim=1)

        def _knn_features(self, x, wf):
            """KNN combined vector (3F): linear + shape + quadratic."""
            eps  = self.eps
            sqwf = wf.sqrt()
            wk   = self._knn_branch_weights()
            lin  = x * sqwf
            mu   = x.mean(1, keepdim=True)
            sd   = x.std(1, keepdim=True).clamp(min=eps)
            shp  = (x - mu) / sd
            sqd  = x * x * sqwf
            return torch.cat([
                wk[0].sqrt() * lin,
                wk[1].sqrt() * shp,
                wk[2].sqrt() * sqd,
            ], dim=1)

        def forward(self, x):
            """
            AVM TRAINING forward pass — Euclidean inverse-distance over the
            global-structure space (tac + bnd + cir) to K*m learnable
            prototypes. This is the training metric only; inference re-scores
            the same prototypes with the QDA posterior (see _avm_proba).
            Embeddings returned are AVM space — used for SupCon and ortho_reg.
            Returns (proba, feat_weights, embeddings).
            """
            wf   = self._feat_weights()
            N    = x.shape[0]
            K, m = self.K, self.m
            tau  = torch.exp(self.log_tau)

            # AVM: global structure space (tac + bnd + cir)
            fx_avm     = self._avm_features(x, wf)                # (N, D)
            proto_flat = self.centroids.reshape(K*m, self.n_feat)
            fy_avm     = self._avm_features(proto_flat, wf)       # (K*m, D)

            sq_x   = (fx_avm**2).sum(1, keepdim=True)
            sq_y   = (fy_avm**2).sum(1)
            dot    = fx_avm @ fy_avm.T
            d_flat = torch.sqrt(torch.clamp(
                sq_x + sq_y[None,:] - 2.0*dot, min=self.eps))    # (N, K*m)
            d      = d_flat.reshape(N, K, m)

            tau_e  = tau.reshape(1,K,1) if (K>1 and tau.dim()>0) else tau
            raw    = (1.0 / (d / tau_e + self.eps)).sum(2)        # (N, K)
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
        # Label smoothing math.
        # Target distribution: q_i = (1-α)*onehot_i + α/K  for all i.
        # CE = -Σ q_i log p_i = (1-α)*NLL + (α/K)*Σ_k -log p_k
        #    = (1-α)*NLL + α * mean_k(-log p_k)
        # NOTE: NLL is class-weighted, the smoothing term is unweighted — left
        # deliberate (smoothing acts as a uniform floor independent of balance).
        smooth_term = -torch.log(probs.clamp(1e-10)).mean(dim=1).mean()
        ce  = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth_term

        # Feature weight regularisation
        _fw = fw if fw is not None else self.net_._feat_weights()
        fw_reg = self.feat_weight_reg * (_fw - 1.0).pow(2).mean()

        # Centroid anchor
        c_anchor = self.centroid_reg * (
            self.net_.centroids - self.net_.centroid_anchor
        ).pow(2).mean()

        # Centroid separation — recomputed every batch and left ATTACHED to the
        # autograd graph so it actually backprops into the prototypes.
        # (Previously this was cached once per epoch and .detach()ed, which made
        # c_sep a constant offset with zero gradient: centroid_sep / intra_sep
        # were silently no-ops. The cost of recomputing is trivial — cdist over
        # K*m prototypes, i.e. tens-to-hundreds of rows, not N.)
        dev = self.net_.centroids.device
        if self.net_.K > 1:
            _K, _m = self.net_.K, self.net_.m
            proto_flat = self.net_.centroids.reshape(_K * _m, self.net_.n_feat)
            c_dists    = torch.cdist(proto_flat, proto_flat, p=2)

            class_idx   = torch.arange(_K, device=dev).repeat_interleave(_m)
            diff_class  = (class_idx.unsqueeze(0) != class_idx.unsqueeze(1)).float()
            upper       = torch.triu(torch.ones_like(diff_class), diagonal=1)
            inter_mask  = diff_class * upper

            if inter_mask.sum() > 0:
                inter = -self.centroid_sep * (
                    c_dists * inter_mask).sum() / inter_mask.sum()
            else:
                inter = torch.zeros((), device=dev)

            intra = torch.zeros((), device=dev)
            if _m > 1 and self.intra_sep > 0.0:
                for k in range(_K):
                    proto_k = self.net_.centroids[k]
                    d_intra = torch.cdist(proto_k, proto_k, p=2)
                    mask_k  = torch.triu(
                        torch.ones(_m, _m, device=dev), diagonal=1)
                    if mask_k.sum() > 0:
                        intra = intra - self.intra_sep * (
                            d_intra * mask_k).sum() / mask_k.sum()

            c_sep = inter + intra
        else:
            c_sep = torch.zeros((), device=dev)

        if self.ordinal and K >= 2:
            emd = self._ordinal_loss(probs, targets)
            ce  = (1.0 - self.ordinal_weight) * ce + self.ordinal_weight * emd

        if self.supcon and embeddings is not None:
            sc  = self._supcon_loss(embeddings, targets)
            ce  = (1.0 - self.supcon_weight) * ce + self.supcon_weight * sc

        if self.ortho_reg > 0.0 and embeddings is not None:
            # embeddings == AVM space: [tac(F), boundary(F or 2F), circular(F)].
            # Decorrelate the THREE semantic branches. The old `width // n_feat`
            # split was wrong under use_dual_boundary: the 2F interleaved
            # boundary block was chopped into two fake F-branches. Collapse the
            # boundary block to its positive faces (even columns) for an F-wide
            # representative, then build a clean 3×3 Gram.
            F = self.net_.n_feat
            tac = embeddings[:, :F]
            if self.net_.use_dual_boundary:
                bnd = embeddings[:, F:3 * F][:, 0::2]   # positive faces → F-dim
                cir = embeddings[:, 3 * F:]
            else:
                bnd = embeddings[:, F:2 * F]
                cir = embeddings[:, 2 * F:]
            chunks = torch.stack([tac, bnd, cir], dim=1)         # (N, 3, F)
            normed = nn.functional.normalize(chunks, dim=2)
            G      = torch.bmm(normed, normed.transpose(1, 2)).mean(0)
            I      = torch.eye(3, device=G.device)
            ortho  = self.ortho_reg * (G - I).pow(2).sum()
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
        f1   = np.zeros(len(labels))
        prec = np.zeros(len(labels))
        rec  = np.zeros(len(labels))
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
        # Balanced class weights, capped. NLLLoss(reduction='mean') divides by
        # the summed target weights, so any global rescale of cw is inert — keep
        # only the cap (which changes the *relative* weights) and drop the old
        # `/ sum * K` no-op.
        cw_t = torch.tensor(cw, dtype=torch.float32).clamp(max=self.weight_cap)

        m = max(1, int(self.n_prototypes))

        # Initialise prototypes — shape (K, m, F).
        # Two modes, controlled by self.boundary_init:
        #
        # (1) jitter init (default) — every prototype starts at the class
        #     mean. m>1 breaks symmetry with small Gaussian jitter so the
        #     NLL gradient + intra_sep pushes them apart during training.
        #
        # (2) boundary-medoid init (opt-in) — prototype 0 at class centroid
        #     (interior anchor), prototypes 1..m-1 at the lowest-positive-
        #     margin training points (frontier defenders). See
        #     _boundary_medoid_init for details.
        rng_proto = np.random.default_rng(
            None if self.random_state is None else self.random_state + 1)

        if self.boundary_init and m > 1:
            proto_init = self._boundary_medoid_init(
                X_tr_n, y_tr, K, m, rng_proto, verbose=self.verbose)
        else:
            class_means = np.stack([
                X_tr_n[y_tr == i].mean(0) for i in range(K)])  # (K, F)
            proto_init  = np.tile(
                class_means[:, None, :], (1, m, 1))             # (K, m, F)
            if m > 1:
                jitter = rng_proto.normal(
                    0, 0.05, proto_init.shape).astype(np.float32)
                proto_init = proto_init + jitter

        centroids = tt(proto_init)    # (K, m, F)

        # Resolve compute device. 'auto' uses CUDA when available, else CPU.
        # Only the PyTorch training loop runs on-device; covariance estimation
        # and FAISS stay on CPU/numpy. Parameters are moved back to CPU before
        # any .numpy() extraction below.
        if self.device == 'auto':
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            dev = torch.device(self.device)
        self.device_ = dev

        self.net_ = self._Net(centroids, K, n_feat, m=m,
                              per_class_tau=self.per_class_tau,
                              use_dual_boundary=self.use_dual_boundary).to(dev)

        opt = optim.Adam(self.net_.parameters(),
                         lr=self.lr, weight_decay=self.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                  opt, T_0=30, T_mult=2, eta_min=1e-5)

        # Pre-bake tensors on-device — no DataLoader overhead
        X_tr_t = tt(X_tr_n).to(dev)
        y_tr_t = tt(y_tr, long=True).to(dev)
        X_va_t = tt(X_va_n).to(dev)
        cw_t   = cw_t.to(dev)
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
            for i in range(n_batch):
                xb = X_shuf[i*bs:(i+1)*bs]
                yb = y_shuf[i*bs:(i+1)*bs]
                opt.zero_grad()
                proba, fw, emb = self.net_(xb)
                loss = self._loss(proba, yb, cw_t, fw=fw,
                                  embeddings=emb if self.supcon else None)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.net_.parameters(), 1.0)
                opt.step()
            sch.step()

            # EMA centroid update — after each epoch, blend learnable centroid
            # positions toward the actual mean of embedded training samples.
            # This stabilises minority class centroids whose backprop gradient
            # signal is weak (few samples → small contribution to NLL loss).
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
                            # Subsample for large classes to avoid O(N*m) cdist.
                            _sub = cls_X
                            if len(cls_X) > 5000:
                                _idx = torch.randperm(
                                    len(cls_X), device=cls_X.device)[:5000]
                                _sub = cls_X[_idx]
                            d_proto = torch.cdist(_sub, protos, p=2)    # (B, m)
                            resp    = torch.softmax(-d_proto, dim=1)    # (B, m)
                            for j in range(m):
                                w    = resp[:, j:j+1]                   # (B, 1)
                                w_sum = w.sum().clamp(min=1e-10)
                                wm   = (w * _sub).sum(0) / w_sum        # (F,)
                                self.net_.centroids.data[cls, j] = (
                                    self.ema_beta       * protos[j]
                                    + (1.0 - self.ema_beta) * wm)
                # EMA discontinuously relocates the centroids parameter; the
                # Adam moment buffers (exp_avg / exp_avg_sq) still describe the
                # pre-EMA position and would be misapplied on the next step,
                # fighting the EMA pull — worst on the minority classes EMA is
                # meant to help. Drop the optimizer state for this parameter so
                # momentum re-accumulates from the new position.
                opt.state.pop(self.net_.centroids, None)

            # Skip validation on non-val epochs.
            # Do NOT increment wait on skipped epochs. The val-epoch branch adds
            # +val_every (5) when val doesn't improve, representing the full
            # window since last val; +1 per skipped epoch was double-counting.
            if (epoch + 1) % val_every != 0 and epoch < self.epochs - 1:
                if self.verbose and (epoch + 1) % 10 == 0:
                    wa = self.net_._avm_branch_weights().detach().cpu().numpy()
                    print(f"  epoch {epoch+1:3d}  (val skipped)"
                          f"  AVM[tac:{wa[0]:.2f} bnd:{wa[1]:.2f} cir:{wa[2]:.2f}]")
                continue

            self.net_.eval()
            with torch.no_grad():
                vp, _, _ = self.net_(X_va_t)
            vf1 = self._val_score(y_va, vp.cpu().numpy().argmax(1))

            if vf1 > best_f1:
                best_f1    = vf1
                best_state = {k: v.clone() for k, v
                              in self.net_.state_dict().items()}
                wait = 0
            else:
                if epoch >= WARMUP:
                    wait += 5   # account for skipped epochs

            if self.verbose and (epoch + 1) % 10 == 0:
                wa = self.net_._avm_branch_weights().detach().cpu().numpy()
                print(f"  epoch {epoch+1:3d}  val_f1={vf1:.4f}"
                      f"  AVM[tac:{wa[0]:.2f} bnd:{wa[1]:.2f} cir:{wa[2]:.2f}]")

            if wait >= self.patience:
                if self.verbose:
                    print(f"  early stopping at epoch {epoch+1}")
                break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.best_val_f1_ = best_f1
        self._n_train_    = len(X_tr)
        self._n_val_      = len(X_va)

        # Extract learned parameters (move to CPU before numpy)
        with torch.no_grad():
            bw = self.net_._branch_weights().cpu().numpy().copy()
            fw = self.net_._feat_weights().cpu().numpy().copy()
            self.branch_weights_ = bw
            self.feat_weights_   = fw
            self.tau_            = torch.exp(self.net_.log_tau).cpu().numpy().copy()

        # Build split numpy vectors — AVM space and KNN space.
        _fw  = fw.copy()
        sqfw = np.sqrt(_fw).astype(np.float32)

        wa = self.net_._avm_branch_weights().detach().cpu().numpy()
        sa_tac = np.float32(np.sqrt(wa[0]))
        sa_b   = np.float32(np.sqrt(wa[1]))
        sa_cir = np.float32(np.sqrt(wa[2]))
        self.avm_branch_weights_ = {'tanh_arccos': float(wa[0]),
                                    'boundary':    float(wa[1]),
                                    'circular':    float(wa[2])}

        # KNN space: configurable branch set, frozen at uniform weight (the
        # KNN space gets no gradient, so its weights are fixed). Each branch
        # contributes F columns; the uniform sqrt-weight keeps the combined
        # vector scale comparable as the branch set grows.
        self.knn_branches_ = list(self.knn_branches)
        n_knn   = max(1, len(self.knn_branches_))
        wk_each = np.float32(np.sqrt(1.0 / n_knn))
        self.knn_branch_weights_ = {b: 1.0 / n_knn for b in self.knn_branches_}
        _rank_sorted = (np.sort(X_tr_n, axis=0).astype(np.float32)
                        if 'rank' in self.knn_branches_ else None)
        _clr_offset  = ((1e-6 - X_tr_n.min(0)).astype(np.float32)
                        if 'clr' in self.knn_branches_ else None)

        # interaction branch (cheap probe): all off-diagonal pairwise products
        # x_i·x_j (i<j) projected by a fixed seeded Gaussian to interaction_dim
        # coords, then z-scored on train so the branch sits on the same scale as
        # the others. A frozen transform (like _rank_sorted) — no learning, no
        # per-step F² blow-up at inference beyond one (n_pairs × dim) matmul.
        # Lives in the KNN space so it's isolated and doubles as a diagnostic:
        # if it carries signal, the KNN head stops being auto-demoted (lambda
        # drops / KNN fusion weight rises).
        _intx = None
        if 'interaction' in self.knn_branches_:
            iu, ju  = np.triu_indices(n_feat, k=1)
            n_pairs = int(len(iu))
            idim    = (n_feat if self.interaction_dim in ('auto', None)
                       else int(self.interaction_dim))
            idim    = max(1, min(idim, max(1, n_pairs)))
            rng_ix  = np.random.default_rng(
                None if self.random_state is None else self.random_state + 7)
            P = (rng_ix.standard_normal((n_pairs, idim)).astype(np.float32)
                 / np.sqrt(max(1, n_pairs)))
            prod_tr = (X_tr_n[:, iu] * X_tr_n[:, ju]).astype(np.float32)
            proj_tr = prod_tr @ P                                    # (N, idim)
            _intx = {'iu': iu, 'ju': ju, 'P': P,
                     'mu': proj_tr.mean(0).astype(np.float32),
                     'sd': proj_tr.std(0).clip(min=1e-6).astype(np.float32),
                     'dim': idim, 'n_pairs': n_pairs}
            self.interaction_info_ = {'n_pairs': n_pairs, 'dim': idim}
            if self.verbose:
                print(f"  interaction branch: {n_pairs} pairs → {idim} dims "
                      f"(fixed random projection, z-scored)")
        self.branch_weights_ = np.array(
            [wa[0], wa[1], wa[2]] + [1.0 / n_knn] * n_knn, dtype=np.float64)

        def _make_avm_combined(xn, batch_size=10_000):
            """AVM space: tac + boundary + circular (3F or 4F with dual)."""
            def _chunk(chunk):
                eps     = 1e-7
                a       = np.arccos(np.clip(chunk, -1+eps, 1-eps))
                tac     = np.tanh(a * 0.8) * sqfw
                norm_sq = (chunk**2).sum(1, keepdims=True)
                if self.use_dual_boundary:
                    bnd_p = norm_sq - 2.0*chunk + 1.0           # (B, F)
                    bnd_n = norm_sq + 2.0*chunk + 1.0           # (B, F)
                    # Interleave [pos_f0, neg_f0, pos_f1, neg_f1, ...] to match
                    # _Net._boundary_dists exactly.
                    bnd_sq = np.stack([bnd_p, bnd_n], axis=2).reshape(
                        chunk.shape[0], -1)                     # (B, 2F)
                    bnd   = np.sqrt(bnd_sq.clip(min=eps))
                else:
                    bnd   = np.sqrt((norm_sq - 2.0*chunk + 1.0).clip(min=eps))
                cir     = np.sqrt(np.clip(1.0-chunk*chunk, eps, None)) * sqfw
                return np.concatenate([
                    sa_tac*tac, sa_b*bnd, sa_cir*cir
                ], axis=1).astype(np.float32)
            if len(xn) <= batch_size:
                return _chunk(xn)
            return np.vstack([_chunk(xn[i:i+batch_size])
                               for i in range(0, len(xn), batch_size)])

        def _knn_branch(name, chunk, eps):
            if name == 'linear':
                return chunk * sqfw
            if name == 'shape':
                mu = chunk.mean(1, keepdims=True)
                sd = chunk.std(1, keepdims=True).clip(min=eps)
                return (chunk - mu) / sd
            if name == 'quadratic':
                return chunk * chunk * sqfw
            if name == 'log':                       # multiplicative geometry
                return np.sign(chunk) * np.log1p(np.abs(chunk)) * sqfw
            if name == 'rank':                      # monotone geometry
                out = np.empty_like(chunk)
                n   = _rank_sorted.shape[0]
                for j in range(chunk.shape[1]):
                    out[:, j] = np.searchsorted(
                        _rank_sorted[:, j], chunk[:, j], side='right') / n
                return out.astype(np.float32)
            if name == 'clr':                       # compositional geometry
                xp = chunk + _clr_offset
                lx = np.log(np.clip(xp, 1e-6, None))
                return (lx - lx.mean(1, keepdims=True)).astype(np.float32)
            if name == 'interaction':               # cross-feature products
                prod = (chunk[:, _intx['iu']] *
                        chunk[:, _intx['ju']]).astype(np.float32)
                proj = prod @ _intx['P']
                return ((proj - _intx['mu']) / _intx['sd']).astype(np.float32)
            raise ValueError(f"unknown KNN branch '{name}'")

        def _make_knn_combined(xn, batch_size=10_000):
            """KNN space: configurable branches, each F-dim, uniform weight."""
            def _chunk(chunk):
                eps = 1e-7
                parts = [wk_each * _knn_branch(b, chunk, eps)
                         for b in self.knn_branches_]
                return np.concatenate(parts, axis=1).astype(np.float32)
            if len(xn) <= batch_size:
                return _chunk(xn)
            return np.vstack([_chunk(xn[i:i+batch_size])
                               for i in range(0, len(xn), batch_size)])

        self._make_avm_combined = _make_avm_combined
        self._make_knn_combined = _make_knn_combined
        self._make_combined     = _make_avm_combined

        # Extract learned prototype positions in combined space
        # centroids: (K, m, F) → flatten to (K*m, F) for _make_combined
        raw_c_n     = self.net_.centroids.detach().cpu().numpy()     # (K, m, F)
        init_c_n    = self.net_.centroid_anchor.cpu().numpy()        # (K, m, F)
        raw_c_flat  = raw_c_n.reshape(K * m, n_feat)
        self.centroids_combined_ = _make_avm_combined(
            raw_c_flat).astype(np.float32)                     # (K*m, D)
        self.n_prototypes_ = m   # actual m used (post-fit)

        # Drift: mean distance each prototype moved from its anchor
        drift_flat = np.sqrt(
            ((raw_c_n - init_c_n) ** 2).sum(-1))              # (K, m)
        self.centroid_drift_ = drift_flat.mean(1)              # (K,) mean over protos
        self.centroid_drift_per_proto_ = drift_flat            # (K, m)

        counts_tr = np.array([(y_tr == i).sum() for i in range(K)],
                              dtype=np.int32)
        self._class_counts_ = counts_tr.copy()
        if self.verbose:
            for i in range(K):
                proto_drifts = "  ".join(
                    f"p{j}={drift_flat[i,j]:.4f}" for j in range(m))
                print(f"  centroid cls{self.classes_[i]}: "
                      f"n={int(counts_tr[i])}  drift=[{proto_drifts}]")

        # Regularised full Mahalanobis covariance in combined space
        #
        # Estimation strategy: Regularised Discriminant Analysis (RDA)
        #   Σ_k = (1-α)*S_k + α*S_pooled
        #   α → 0: full per-class QDA   α → 1: shared pooled (LDA)   α=0.5 blend
        # Then ridge regularise:  Σ_k_reg = (1-r)*Σ_k + r*I*mean(diag(Σ_k))
        # Cholesky used for stable inversion AND for the log-determinant that
        # feeds the QDA volume term in _avm_proba.
        if self.mahalanobis:
            X_tr_c = _make_avm_combined(X_tr_n)   # AVM space for Mahal

            # Per-column standardisation BEFORE covariance estimation. The
            # combined AVM vector stacks branches on very different scales
            # (boundary ∝ √F dominates tac∈tanh, cir∈[0,1]). Mahalanobis
            # distance itself is invariant to a fixed diagonal rescale, but the
            # ridge term (r·mean(diag)·I) and the RDA pooling are NOT — without
            # this the ridge under-regularises the small-scale branches. Store
            # the scale; apply the SAME standardisation to the query and the
            # centroids inside _avm_proba.
            self.avm_scale_ = X_tr_c.std(0).clip(min=1e-6).astype(np.float32)
            X_tr_c = X_tr_c / self.avm_scale_

            D = X_tr_c.shape[1]

            # Pooled covariance — weighted sum of per-class covariances
            S_pooled = np.zeros((D, D), dtype=np.float64)
            for i in range(K):
                cls_pts = X_tr_c[y_tr == i].astype(np.float64)
                if len(cls_pts) < 2:
                    continue
                c = cls_pts - cls_pts.mean(0, keepdims=True)
                S_pooled += (c.T @ c)
            S_pooled /= (len(X_tr_c) - K)   # unbiased pooled estimate

            # Storage: (K*m, D, D) cov_inv + (K*m,) log-determinants — one per
            # prototype (n_prototypes>1 spreads prototypes via intra_sep to
            # capture sub-clusters with DIFFERENT shapes, so each needs its own
            # covariance estimated from its soft-assigned points).
            self.cov_inv_    = np.zeros((K * m, D, D), dtype=np.float32)
            self.cov_logdet_ = np.zeros(K * m, dtype=np.float32)
            alpha            = float(self.mahal_alpha)
            ridge            = float(self.mahal_reg)

            # Soft assignments in normalised raw space use the learned prototype
            # positions. For m=1 weights collapse to the class membership mask.
            X_tr_n_t = torch.tensor(X_tr_n, dtype=torch.float32)

            for i in range(K):
                cls_mask = (y_tr == i)
                cls_pts  = X_tr_c[cls_mask].astype(np.float64)
                n_k      = len(cls_pts)

                if n_k < 2:
                    # Degenerate — replicate pooled to all m prototypes
                    for j in range(m):
                        (self.cov_inv_[i * m + j],
                         self.cov_logdet_[i * m + j]) = self._invert_cov(
                            S_pooled, ridge, D)
                    continue

                if m == 1:
                    # Single prototype: all points belong to it
                    c   = cls_pts - cls_pts.mean(0, keepdims=True)
                    S_k = (c.T @ c) / (n_k - 1)
                    S_blend = (1.0 - alpha) * S_k + alpha * S_pooled
                    (self.cov_inv_[i],
                     self.cov_logdet_[i]) = self._invert_cov(S_blend, ridge, D)
                else:
                    # Per-prototype: soft-assign by distance in raw space,
                    # then weight covariance estimate accordingly.
                    cls_pts_raw = X_tr_n_t[torch.tensor(cls_mask)]   # (n_k, F)
                    protos_raw  = torch.tensor(raw_c_n[i],
                                                dtype=torch.float32)  # (m, F)
                    d_proto = torch.cdist(cls_pts_raw, protos_raw, p=2)
                    resp    = torch.softmax(-d_proto, dim=1).numpy()  # (n_k, m)

                    for j in range(m):
                        w     = resp[:, j].astype(np.float64)          # (n_k,)
                        w_sum = max(w.sum(), 1e-10)
                        # Weighted mean
                        mu_j  = (w[:, None] * cls_pts).sum(0) / w_sum
                        # Weighted covariance (Bessel-style)
                        c     = cls_pts - mu_j
                        S_j   = (w[:, None] * c).T @ c / w_sum
                        # Effective sample size correction
                        ess   = (w.sum() ** 2) / (w ** 2).sum().clip(min=1e-10)
                        if ess > 1:
                            S_j *= ess / (ess - 1)
                        S_blend = (1.0 - alpha) * S_j + alpha * S_pooled
                        (self.cov_inv_[i * m + j],
                         self.cov_logdet_[i * m + j]) = self._invert_cov(
                            S_blend, ridge, D)

            if self.verbose:
                print(f"  Mahalanobis: per-prototype RDA "
                      f"(alpha={alpha:.2f} ridge={ridge:.0e} "
                      f"D={D} K*m={K*m}, standardised + logdet)")
        else:
            self.cov_inv_    = None
            self.cov_logdet_ = None
            self.avm_scale_  = None

        # ── KNN head ────────────────────────────────────────────────────
        self.active_heads_ = ['avm']
        self.lambda_ = self.lam_floor
        self.index_  = None
        _build_knn = ('knn' in set(self.heads)) and _HAS_FAISS
        if not _build_knn:
            if 'knn' in set(self.heads) and not _HAS_FAISS and self.verbose:
                print("  faiss not installed — KNN head disabled")
            self.lambda_ = 1.0
        else:
            X_tr_c  = _make_knn_combined(X_tr_n)  # KNN space for FAISS
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

            # Lambda selection. When extra heads (fisher/geodesic) are
            # configured, _fit_fusion owns head combination and self.lambda_ is
            # never read — skip the otherwise-dead scalar/gate search entirely.
            extra_heads = bool(set(self.heads) & {'fisher', 'geodesic'})

            if self.entropy_lambda:
                # AVM confidence gate — no training needed.
                # lambda(x) = lam_confident - (lam_confident-lam_uncertain)*H(x)
                # High AVM entropy (uncertain) → low lambda (trust KNN more).
                self.lambda_ = None   # signals entropy gate in predict_proba
                if self.verbose:
                    print(f"  lambda mode=entropy  "
                          f"confident={self.lam_confident}  "
                          f"uncertain={self.lam_uncertain}")

            elif extra_heads:
                self.lambda_ = self.lam_floor
                if self.verbose:
                    print("  lambda: skipped (extra heads → fusion owns "
                          "head combination)")

            else:
                # Compare scalar search vs confidence gate on the same AVM
                # probabilities predict_proba produces.
                X_va_avm = _make_avm_combined(X_va_n)
                avm_val  = self._avm_proba(X_va_avm)

                X_va_knn = _make_knn_combined(X_va_n)
                knn_val  = self._knn_proba(X_va_knn)

                # Option A — scalar grid search (linspace keeps lam=1.0 in grid)
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

                # Option B — confidence-weighted gate (AVM vs KNN)
                lam_conf, _ = self._lambda_gate(avm_val, knn_val)
                p_conf      = lam_conf[:,None]*avm_val + (1-lam_conf[:,None])*knn_val
                conf_f1 = self._val_score(y_va, p_conf.argmax(1))

                if conf_f1 >= best_lam_f1:
                    self.lambda_ = None   # None = use confidence gate
                    if self.verbose:
                        print(f"  lambda: confidence gate  val_f1={conf_f1:.4f}"
                              f" (scalar={best_lam_f1:.4f})")
                else:
                    self.lambda_ = best_lam
                    if self.verbose:
                        print(f"  lambda: scalar={best_lam:.2f}"
                              f"  val_f1={best_lam_f1:.4f}"
                              f" (conf={conf_f1:.4f})")

        if self.index_ is not None:
            self.active_heads_.append('knn')

        # ── Fisher head (supervised between-class directions) ────────────
        self.fisher_ = None
        if 'fisher' in set(self.heads):
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            self.fisher_ = LinearDiscriminantAnalysis(
                solver='eigen', shrinkage='auto').fit(X_tr_n, y_tr)
            self.active_heads_.append('fisher')
            if self.verbose:
                print(f"  Fisher head: {len(self.classes_)-1} LDA directions")

        # ── Geodesic head (intrinsic manifold geometry) ─────────────────
        self.geo_ = None
        if 'geodesic' in set(self.heads):
            from sklearn.manifold import Isomap
            if len(X_tr_n) > self.geodesic_max_fit:
                sub = rng.choice(len(X_tr_n), self.geodesic_max_fit,
                                 replace=False)
            else:
                sub = np.arange(len(X_tr_n))
            nn_ = max(2, min(self.geodesic_neighbors, len(sub) - 1))
            nc = max(1, min(self.geodesic_components, n_feat))
            self.geo_ = Isomap(n_neighbors=nn_, n_components=nc).fit(X_tr_n[sub])
            emb = self.geo_.transform(X_tr_n).astype(np.float64)
            self.geo_centroids_ = np.stack(
                [emb[y_tr == c].mean(0) if (y_tr == c).any()
                 else emb.mean(0) for c in range(K)])
            self.active_heads_.append('geodesic')
            if self.verbose:
                print(f"  Geodesic head: Isomap nn={nn_} dims={nc} "
                      f"(fit on {len(sub)} pts)")

        # ── Fusion scheme ───────────────────────────────────────────────
        ah = set(self.active_heads_)
        if ah == {'avm'}:
            self.fusion_ = {'mode': 'avm_only'}
        elif ah == {'avm', 'knn'}:
            self.fusion_ = {'mode': 'legacy'}      # uses self.lambda_, unchanged
        else:
            P_val = self._collect_head_probas(X_va_n)
            self.fusion_ = self._fit_fusion(P_val, y_va)
            if self.verbose:
                print(f"  fusion: {self.active_heads_} -> {self.fusion_}")

        return self

    # ── inference helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _boundary_medoid_init(X_tr_n, y_tr, K, m, rng, verbose=False):
        """
        Boundary-medoid prototype placement (opt-in via boundary_init=True).

          proto[k, 0]     = class centroid (interior anchor — captures bulk).
          proto[k, 1..]   = lowest-positive-margin training points in class k.

        Margin = d(nearest opposing class centroid) - d(own class centroid),
        measured in Euclidean normalised space. Negative margins are excluded
        (likely label noise). Falls back to jitter when a class has too few
        points or too few positive-margin candidates.
        """
        n_feat = X_tr_n.shape[1]
        class_means = np.stack([X_tr_n[y_tr == i].mean(0) for i in range(K)])
        proto = np.tile(class_means[:, None, :], (1, m, 1)).astype(np.float32)

        if m == 1:
            return proto

        # All-pairs distance via norm-expansion (avoids the (N,K,F) tensor).
        d_all = np.sqrt(np.clip(
            (X_tr_n ** 2).sum(1, keepdims=True)
            + (class_means ** 2).sum(1)[None, :]
            - 2.0 * X_tr_n @ class_means.T,
            0.0, None))                                          # (N, K)

        fallback_classes = []
        for k in range(K):
            idx_k = np.where(y_tr == k)[0]
            n_k   = len(idx_k)

            if n_k < m:
                proto[k, 1:] = class_means[k] + rng.normal(
                    0, 0.05, (m - 1, n_feat)).astype(np.float32)
                fallback_classes.append(int(k))
                continue

            d_own   = d_all[idx_k, k]
            d_other = np.delete(d_all[idx_k], k, axis=1).min(1)
            margin  = d_other - d_own                            # (n_k,)

            pos = margin > 0
            if pos.sum() >= m - 1:
                cand_idx, cand_mgn = idx_k[pos], margin[pos]
            else:
                cand_idx, cand_mgn = idx_k, margin
                fallback_classes.append(int(k))

            order = np.argsort(cand_mgn)[:m - 1]
            proto[k, 1:] = X_tr_n[cand_idx[order]]

        # Tiny jitter — break exact duplicates, helps gradient symmetry
        proto = proto + rng.normal(0, 0.005, proto.shape).astype(np.float32)

        if verbose:
            if fallback_classes:
                print(f"  boundary-init: jitter fallback for classes "
                      f"{fallback_classes} (insufficient positive-margin points)")
            else:
                print(f"  boundary-init: clean placement for all {K} classes")

        return proto.astype(np.float32)

    @staticmethod
    def _invert_cov(S, ridge, D):
        """Ridge-regularise and Cholesky-invert a covariance matrix, returning
        (Sigma_inv, log|Sigma|). The log-determinant is read off the Cholesky
        factor (2·Σ log diag L) at no extra cost and feeds the QDA volume term
        −0.5·log|Σ_k| in _avm_proba. Cholesky is fast on well-conditioned
        matrices; on failure we escalate the ridge (10x, up to 3x) and retry
        before paying for slogdet + pinv."""
        scale = np.diag(S).mean().clip(min=1e-10)
        r = float(ridge)
        for _ in range(3):
            S_reg = (1.0 - r) * S + r * scale * np.eye(D)
            try:
                L      = np.linalg.cholesky(S_reg)
                L_inv  = np.linalg.solve(L, np.eye(D))
                logdet = 2.0 * np.log(np.diag(L)).sum()
                return (L_inv.T @ L_inv).astype(np.float32), np.float32(logdet)
            except np.linalg.LinAlgError:
                r = min(r * 10.0, 1.0)
        # Last resort — pseudo-inverse + slogdet of the most-regularised blend
        S_reg     = (1.0 - r) * S + r * scale * np.eye(D)
        _, logdet = np.linalg.slogdet(S_reg)
        return np.linalg.pinv(S_reg).astype(np.float32), np.float32(logdet)

    def _avm_proba(self, combined, batch_size=8192):
        """
        QDA posterior over K classes, each an equal-weight mixture of its m
        prototype Gaussians:

            log p(k|x) = log π_k
                       + logsumexp_{p∈k}[ −0.5·log|Σ_p| − 0.5·d²_M(x, c_p) ]
                       + const   (−0.5·D·log 2π and −log m cancel in softmax_k)

        then softmax over k. With cov_inv_ set, d²_M is the full per-prototype
        Mahalanobis distance on STANDARDISED combined features (avm_scale_);
        with cov_inv_ None it degenerates to −0.5·‖x−c‖² (identity-covariance /
        spherical QDA). log|Σ_p| is the volume term the old inverse-distance
        score dropped — without it large-covariance classes were systematically
        over-predicted.

        logdet_weight scales JUST the volume term (qda mode only):
          1.0 = full QDA (statistically correct posterior).
          0.0 = Mahalanobis distance + prior softmax, no volume term.
        Distance and prior are untouched by the dial.

        avm_score selects the decision form:
          'qda'              softmax over the Gaussian discriminant above —
                             sharp, light-tailed, statistically correct.
          'inverse_distance' the original AVM vote Σ_p 1/d_M, renormalised —
                             flat, heavy-tailed, better calibrated for the
                             downstream entropy gates / fusion. Recovers the
                             pre-QDA behaviour (on Mahalanobis distance); the
                             volume term has no analogue here and is ignored.

        prior_weight scales the empirical class prior in BOTH modes:
          1.0 = full π_k (qda: +log π_k in the logit; invdist: ×π_k).
          0.0 = no prior — raises minority recall / macro-F1 on imbalanced
                data at the cost of strict posterior correctness.
        The pre-QDA scoring is recovered with avm_score='inverse_distance',
        prior_weight=0.0.

        Centring note: the Gaussian component is centred on the LEARNED
        prototype c_p (centroids_combined_); Σ_p was estimated about the soft-
        assigned empirical mean. For m=1 these coincide; for m>1 (e.g. boundary-
        init frontier prototypes) they differ by design — the prototype defines
        where the component sits, Σ_p its shape.

        Processed in row-batches so peak memory is O(batch_size·D).
        """
        K   = len(self.classes_)
        m   = self.n_prototypes_
        N   = combined.shape[0]
        P   = K * m

        priors    = self._class_counts_ / self._class_counts_.sum()
        log_prior = np.log(np.clip(priors, 1e-12, None))
        pw        = float(getattr(self, 'prior_weight', 1.0))

        # ── squared distance per prototype (Mahalanobis if cov else Euclidean)
        #    + per-prototype volume offset (0 when no covariance) ─────────────
        if self.cov_inv_ is not None:
            scale  = self.avm_scale_.astype(np.float64)             # (D,)
            cov64  = self.cov_inv_.astype(np.float64)               # (P, D, D)
            cen64  = self.centroids_combined_.astype(np.float64) / scale  # (P,D)
            w_ld   = float(getattr(self, 'logdet_weight', 1.0))     # volume dial
            ld_eff = w_ld * self.cov_logdet_.astype(np.float64)     # (P,)
            d2     = np.empty((N, P), dtype=np.float64)
            for s in range(0, N, batch_size):
                e     = min(s + batch_size, N)
                chunk = combined[s:e].astype(np.float64) / scale    # (B, D) std
                for p in range(P):
                    diff = chunk - cen64[p]                         # (B, D)
                    tmp  = diff @ cov64[p]                          # (B, D)
                    d2[s:e, p] = np.maximum((tmp * diff).sum(1), 0.0)
        else:
            a_sq   = (combined ** 2).sum(1, keepdims=True)
            b_sq   = (self.centroids_combined_ ** 2).sum(1)
            ab     = combined @ self.centroids_combined_.T
            d2     = np.maximum(a_sq + b_sq[None, :] - 2.0 * ab, 0.0)
            ld_eff = np.zeros(P, dtype=np.float64)

        # ── decision form ──────────────────────────────────────────────────
        if getattr(self, 'avm_score', 'qda') == 'inverse_distance':
            # Heavy-tailed inverse-distance vote over prototypes (the original
            # AVM form), now on the Mahalanobis distance. Flat, well-calibrated
            # for the downstream entropy gates. logdet_weight has no clean
            # analogue here and is ignored; prior_weight applies as π_k^pw.
            d_km = np.sqrt(d2).reshape(N, K, m)
            raw  = (1.0 / (d_km + 1e-10)).sum(2)                    # (N, K)
            if pw != 0.0:
                raw = raw * (priors[None, :] ** pw)
            return (raw / raw.sum(1, keepdims=True)).astype(np.float32)

        # QDA posterior — equal-weight prototype mixture, prior in the logit.
        log_comp = (-0.5 * (d2 + ld_eff[None, :])).reshape(N, K, m)
        cmax = log_comp.max(2)                                      # (N, K)
        lse  = cmax + np.log(np.exp(log_comp - cmax[..., None]).sum(2))
        logits = lse + pw * log_prior[None, :]                      # (N, K)
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        return (e / e.sum(1, keepdims=True)).astype(np.float32)

    def _knn_proba(self, combined, search_batch=50_000):
        """
        KNN inverse-distance vote. Vectorized scatter.
        """
        if self.index_ is None:
            raise RuntimeError("FAISS index not built.")
        n_test = combined.shape[0]
        K      = len(self.classes_)
        proba  = np.zeros((n_test, K), dtype=np.float32)
        for start in range(0, n_test, search_batch):
            end    = min(start + search_batch, n_test)
            dist, idx = self.index_.search(combined[start:end], self.k)
            w = 1.0 / (dist + 1e-10)
            w = w / w.sum(1, keepdims=True)                  # (B, k)
            labels = self.train_class_idx_[idx]               # (B, k)
            B = end - start
            batch_p = np.zeros((B, K), dtype=np.float32)
            for ki in range(self.k):
                batch_p[np.arange(B), labels[:, ki]] += w[:, ki]
            proba[start:end] = batch_p
        return proba

    def _lambda_gate(self, avm, knn):
        """
        Confidence-weighted gate: AVM vs KNN.
        lambda = conf_AVM / (conf_AVM + conf_KNN).
        """
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

    # ── head probabilities & N-head fusion ─────────────────────────────────

    def _fisher_proba(self, Xn):
        return self.fisher_.predict_proba(Xn).astype(np.float64)

    def _geodesic_proba(self, Xn):
        emb = self.geo_.transform(Xn).astype(np.float64)          # (N, nc)
        c   = self.geo_centroids_                                  # (K, nc)
        a   = (emb ** 2).sum(1, keepdims=True)
        b   = (c ** 2).sum(1)
        d   = np.sqrt(np.maximum(a + b[None, :] - 2.0 * emb @ c.T, 0.0))
        raw = 1.0 / (d + 1e-10)
        return raw / raw.sum(1, keepdims=True)

    def _head_proba(self, name, Xn):
        if name == 'avm':
            return self._avm_proba(self._make_avm_combined(Xn))
        if name == 'knn':
            return self._knn_proba(self._make_knn_combined(Xn))
        if name == 'fisher':
            return self._fisher_proba(Xn)
        if name == 'geodesic':
            return self._geodesic_proba(Xn)
        raise ValueError(f"unknown head '{name}'")

    def _collect_head_probas(self, Xn):
        return {h: self._head_proba(h, Xn) for h in self.active_heads_}

    def _conf_weights(self, probas):
        """Per-sample confidence (1 - normalised entropy) per head, normalised."""
        K = len(self.classes_); logK = np.log(K + 1e-10)
        C = np.stack([np.clip(
            1.0 + (p * np.log(np.clip(p, 1e-10, None))).sum(1) / logK,
            1e-6, None) for p in probas], axis=1)          # (N, n_heads)
        return C / C.sum(1, keepdims=True)

    def _fit_fusion(self, P_val, y_va, floor=0.05, shrink=0.5, margin=1e-3):
        """Pick N-head fusion on validation, regularised against overfitting a
        small/imbalanced val split: floored + shrunk static weights, departing
        from uniform only past a margin, preferring the parameter-free
        confidence gate over fitted weights."""
        heads = self.active_heads_
        probs = [P_val[h] for h in heads]
        n = len(heads)
        uni = np.full(n, 1.0 / n)

        def score(weights):
            p = sum(wi * pi for wi, pi in zip(weights, probs))
            return self._val_score(y_va, p.argmax(1))

        uni_f1 = score(uni)

        # static coordinate ascent with a per-head floor (no head zeroed)
        w, best = uni.copy(), uni_f1
        for _ in range(4):
            for i in range(n):
                cur, cur_w = best, w
                for g in np.linspace(floor, 1.0, 11):
                    cand = w.copy(); cand[i] = g
                    cand = np.clip(cand, floor, None)
                    cand = cand / cand.sum()
                    sc = score(cand)
                    if sc > cur:
                        cur, cur_w = sc, cand
                w, best = cur_w, cur
        static_w  = (1.0 - shrink) * w + shrink * uni
        static_f1 = score(static_w)

        # confidence gate (per-sample, no fitted weights)
        W = self._conf_weights(probs)
        p_conf  = sum(W[:, i:i+1] * probs[i] for i in range(n))
        conf_f1 = self._val_score(y_va, p_conf.argmax(1))

        # robustness-ordered selection: uniform < confidence < static
        mode, mode_f1 = 'uniform', uni_f1
        if conf_f1 > mode_f1 + margin:
            mode, mode_f1 = 'confidence', conf_f1
        if static_f1 > mode_f1 + margin:
            mode, mode_f1 = 'static', static_f1

        if mode == 'confidence':
            return {'mode': 'confidence', 'heads': heads, 'val_f1': float(conf_f1)}
        weights = static_w if mode == 'static' else uni
        return {'mode': 'static', 'heads': heads,
                'weights': {h: float(wi) for h, wi in zip(heads, weights)},
                'val_f1': float(mode_f1)}

    def _fuse(self, P):
        if self.fusion_['mode'] == 'avm_only':
            return P['avm']
        heads = self.fusion_['heads']
        probs = [P[h] for h in heads]
        if self.fusion_['mode'] == 'static':
            w = self.fusion_['weights']
            return sum(w[h] * P[h] for h in heads)
        W = self._conf_weights(probs)                      # confidence
        return sum(W[:, i:i+1] * probs[i] for i in range(len(heads)))

    def _apply_gravity(self, proba):
        """Decision-stage prior tempering ('centroid gravity'):
        p'(k|x) ∝ p(k|x) · π_k^α, i.e. log p + α·log π_k, renormalised.
        With the QDA prior now built into _avm_proba, this TEMPERS that prior
        rather than adding a second one — keep prior_temp=0 (default) for a
        single, correctly-weighted prior; >0 deliberately over-weights it."""
        a = float(getattr(self, 'prior_temp', 0.0))
        if a == 0.0:
            return proba
        counts = getattr(self, '_class_counts_', None)
        if counts is None:
            return proba
        log_prior = np.log(np.clip(counts / counts.sum(), 1e-12, None))
        logp = np.log(np.clip(proba, 1e-12, None)) + a * log_prior[None, :]
        logp -= logp.max(1, keepdims=True)
        e = np.exp(logp)
        return (e / e.sum(1, keepdims=True)).astype(proba.dtype)

    def predict_proba(self, X):
        check_is_fitted(self, 'centroids_combined_')
        X        = np.asarray(X, dtype=np.float32)
        Xn       = self._norm_apply(X)

        mode = getattr(self, 'fusion_', {'mode': 'legacy'})['mode']
        if mode not in ('legacy', 'avm_only'):
            return self._apply_gravity(self._fuse(self._collect_head_probas(Xn)))

        Xn_avm = self._make_avm_combined(Xn)
        avm    = self._avm_proba(Xn_avm)

        if mode == 'avm_only' or self.index_ is None or (
                self.lambda_ is not None and self.lambda_ >= 1.0):
            return self._apply_gravity(avm)

        knn = self._knn_proba(self._make_knn_combined(Xn))  # KNN space

        if self.entropy_lambda:
            K   = len(self.classes_)
            H   = -(avm * np.log(avm.clip(1e-10))).sum(1) / np.log(K)
            lam = (self.lam_confident
                   - (self.lam_confident - self.lam_uncertain) * H)
            lam = lam.clip(self.lam_uncertain, self.lam_confident)
            proba = lam[:,None]*avm + (1-lam[:,None])*knn
        elif self.lambda_ is None:
            lam, _ = self._lambda_gate(avm, knn)
            proba = lam[:,None]*avm + (1-lam[:,None])*knn
        else:
            proba = self.lambda_*avm + (1.0-self.lambda_)*knn
        return self._apply_gravity(proba)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_with_uncertainty(self, X):
        """
        Returns (predictions, proba, disagreement_scores).
        Disagreement: confident heads predicting different classes.
        """
        check_is_fitted(self, 'centroids_combined_')
        X  = np.asarray(X, dtype=np.float32)
        Xn = self._norm_apply(X)
        mode = getattr(self, 'fusion_', {'mode': 'legacy'})['mode']

        if mode == 'legacy' and self.index_ is not None:
            avm = self._avm_proba(self._make_avm_combined(Xn))
            knn = self._knn_proba(self._make_knn_combined(Xn))
            lam, disagree = self._lambda_gate(avm, knn)
            proba = lam[:, None] * avm + (1 - lam[:, None]) * knn
            return self.classes_[proba.argmax(1)], proba, disagree

        P     = self._collect_head_probas(Xn)
        proba = P['avm'] if mode == 'avm_only' else self._fuse(P)
        pred  = proba.argmax(1)
        heads = self.active_heads_
        if len(heads) > 1:
            W = self._conf_weights([P[h] for h in heads])
            disagree = np.zeros(len(X), dtype=np.float64)
            for i, h in enumerate(heads):
                disagree += W[:, i] * (P[h].argmax(1) != pred)
        else:
            disagree = 1.0 - proba.max(1)
        return self.classes_[pred], proba, disagree.astype(np.float32)

    def class_distribution(self, as_dict=False):
        """
        Training class distribution as percentages.
        """
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

    def report(self, X, y, as_dict=False):
        """
        Per-class precision / recall / F1 / support + confusion matrix from the
        fitted model — the targeted diagnostic for an imbalanced squeeze (shows
        exactly which class is dragging macro-F1, e.g. a minority class with
        high precision but floored recall).

        All metrics are derived from a single confusion matrix (one predict
        pass), matching _val_score. Rows of the printed matrix are TRUE classes,
        columns are PREDICTED; the lowest-F1 class is flagged.

        as_dict=True returns
          {'per_class': {label: {'precision','recall','f1','support'}},
           'accuracy', 'macro_f1', 'weighted_f1',
           'confusion': ndarray (K,K), 'labels': classes_}
        """
        from sklearn.metrics import confusion_matrix
        check_is_fitted(self, 'centroids_combined_')
        y_pred = self.predict(np.asarray(X, dtype=np.float32))
        y_true = np.asarray(y)

        labels = self.classes_
        K      = len(labels)
        cm     = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
        tp     = np.diag(cm)
        support = cm.sum(1)          # true count per class
        pred_s  = cm.sum(0)          # predicted count per class
        total   = cm.sum()

        prec = np.zeros(K); rec = np.zeros(K); f1 = np.zeros(K)
        np.divide(tp, pred_s, out=prec, where=pred_s > 0)
        np.divide(tp, support, out=rec, where=support > 0)
        denom = prec + rec
        np.divide(2 * prec * rec, denom, out=f1, where=denom > 0)

        acc      = tp.sum() / (total + 1e-10)
        macro    = f1.mean()
        weighted = (f1 * support).sum() / (total + 1e-10)

        if as_dict:
            return {
                'per_class': {
                    labels[i]: {'precision': float(prec[i]),
                                'recall':    float(rec[i]),
                                'f1':        float(f1[i]),
                                'support':   int(support[i])}
                    for i in range(K)},
                'accuracy':    float(acc),
                'macro_f1':    float(macro),
                'weighted_f1': float(weighted),
                'confusion':   cm.astype(int),
                'labels':      labels,
            }

        worst = int(np.argmin(f1))
        lab_w = max(8, max(len(str(c)) for c in labels))
        lines = [
            f"  {'class':<{lab_w}}  prec    recall  f1      support",
            f"  {'-'*lab_w}  ------  ------  ------  -------",
        ]
        for i in range(K):
            flag = "  ← lowest F1" if i == worst and K > 1 else ""
            lines.append(
                f"  {str(labels[i]):<{lab_w}}  {prec[i]:.4f}  {rec[i]:.4f}"
                f"  {f1[i]:.4f}  {int(support[i]):>7d}{flag}")
        lines += [
            f"  {'-'*lab_w}  ------  ------  ------  -------",
            f"  {'accuracy':<{lab_w}}                  {acc:.4f}  {int(total):>7d}",
            f"  {'macro':<{lab_w}}                  {macro:.4f}",
            f"  {'weighted':<{lab_w}}                  {weighted:.4f}",
            "",
            "  Confusion (rows=true, cols=pred):",
        ]
        cell = max(6, max(len(str(c)) for c in labels) + 1)
        header = " " * (lab_w + 4) + "".join(
            f"{str(c):>{cell}}" for c in labels)
        lines.append(header)
        for i in range(K):
            row = "".join(f"{int(cm[i, j]):>{cell}d}" for j in range(K))
            lines.append(f"  {str(labels[i]):<{lab_w}}  {row}")
        return "\n".join(lines)

    def summary(self, feature_names=None):
        """Print a human-readable model summary after fitting."""
        check_is_fitted(self, 'branch_weights_')
        names = (list(feature_names) if feature_names is not None
                 else [f"f{i}" for i in range(len(self.feat_weights_))])
        bw  = self.branch_weights_
        lam = getattr(self, 'lambda_', self.lam_floor)
        tau = self.tau_
        tau_str = (f"{float(tau):.3f}" if np.ndim(tau) == 0 or np.size(tau) == 1
                   else "  ".join(f"cls{i}:{v:.3f}" for i, v in enumerate(tau)))
        lam_str = ("[confidence gate]" if lam is None else
                   "[entropy gate]"    if self.entropy_lambda else
                   f"{lam:.3f}")
        K    = len(self.classes_)
        n_tr = getattr(self, '_n_train_', '?')
        n_va = getattr(self, '_n_val_',   '?')
        dist = self.class_distribution()
        lines = [
            f"  Classes ({K}): {dist}",
            f"  Train / val  : {n_tr} / {n_va}",
            "",
            f"  Lambda  : {lam_str}",
            f"  Tau     : {tau_str}  (training-only)",
            f"  Val F1  : {getattr(self, 'best_val_f1_', float('nan')):.4f}",
            "",
            f"  Discriminant: QDA posterior (log π_k − 0.5·log|Σ_k| − 0.5·d²_M)",
            f"  AVM score   : {self.avm_score}  prior_weight={self.prior_weight}",
            f"  Mahalanobis : {'full RDA (standardised)' if self.mahalanobis else 'off (spherical QDA)'}"
            f"  (alpha={self.mahal_alpha}  ridge={self.mahal_reg:.0e})",
            f"  Logdet wt   : {self.logdet_weight}  (1=full QDA volume term, 0=distance+prior)",
            f"  Prior temp  : {self.prior_temp}  (0 = single built-in prior)",
            f"  Ordinal EMD : {self.ordinal}  (weight={self.ordinal_weight})",
            f"  SupCon      : {self.supcon}"
            f"  (weight={self.supcon_weight}  temp={self.supcon_temp})",
            f"  EMA         : {self.ema_centroids}  (beta={self.ema_beta})",
            f"  Dual bnd    : {self.use_dual_boundary}",
            f"  Boundary init: {self.boundary_init}",
            f"  n_prototypes: {getattr(self, 'n_prototypes_', self.n_prototypes)}"
            f"  per_class_τ: {self.per_class_tau}",
            f"  ortho_reg   : {self.ortho_reg}",
            f"  Heads       : {getattr(self, 'active_heads_', ['avm','knn'])}"
            f"  fusion={getattr(self, 'fusion_', {}).get('mode', 'legacy')}",
            "",
            f"  AVM branches (global view):",
        ]
        avmw = getattr(self, 'avm_branch_weights_',
                       {'tanh_arccos': bw[0], 'boundary': bw[1],
                        'circular': bw[2]})
        for nm, val in avmw.items():
            lines.append(f"    {nm:<12}: {val:.4f}")
        lines.append("  KNN branches (frozen local view):")
        for nm, val in getattr(self, 'knn_branch_weights_',
                               {'linear': bw[3] if len(bw) > 3 else 0.0}).items():
            lines.append(f"    {nm:<12}: {val:.4f}")
        if self.fusion_.get('mode') == 'static':
            lines.append("  Fusion weights:")
            for h, w in self.fusion_['weights'].items():
                lines.append(f"    {h:<12}: {w:.4f}")
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
            per_proto = getattr(self, 'centroid_drift_per_proto_', None)
            if per_proto is not None and per_proto.shape[1] > 1:
                lines.append("  Per-prototype drift:")
                for ci, cls in enumerate(self.classes_):
                    row = "  ".join(f"p{j}={per_proto[ci, j]:.4f}"
                                    for j in range(per_proto.shape[1]))
                    lines.append(f"    class {cls}: {row}")
        if lam is not None and not self.entropy_lambda:
            lines.append("")
            lines.append(f"  Lambda search: floor={self.lam_floor}"
                         f"  chosen={lam_str}")
        return "\n".join(lines)
