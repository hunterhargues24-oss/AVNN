#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoLambdaAVNN – A unified four‑branch classifier with automatic λ tuning.
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
    def __init__(self,
                 k=5,
                 lr=1e-3,
                 epochs=300,
                 batch_size=64,
                 patience=60,
                 val_fraction=0.15,
                 label_smoothing=0.05,
                 weight_cap=1.5,
                 weight_decay=1e-4,
                 feat_weight_reg=0.05,
                 gravity=0.0,
                 gravity_cap=1.5,
                 lam_init=0.7,
                 use_ivf=True,
                 nlist=1000,
                 nprobe=10,
                 auto_lambda=True,
                 random_state=None,
                 verbose=False):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.label_smoothing = label_smoothing
        self.weight_cap = weight_cap
        self.weight_decay = weight_decay
        self.feat_weight_reg = feat_weight_reg
        self.gravity = gravity
        self.gravity_cap = gravity_cap
        self.lam_init = lam_init
        self.use_ivf = use_ivf
        self.nlist = nlist
        self.nprobe = nprobe
        self.auto_lambda = auto_lambda
        self.random_state = random_state
        self.verbose = verbose

    # ----------------------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------------------
    def _norm_fit(self, X):
        self.mn_ = X.min(0)
        self.rng_ = np.maximum(X.max(0) - self.mn_, 1e-10)

    def _norm_apply(self, X):
        return (-1.0 + 2.0 * (X - self.mn_) / self.rng_).astype(np.float32)

    # ----------------------------------------------------------------------
    # Internal PyTorch model (pure AVM)
    # ----------------------------------------------------------------------
    class _Net(nn.Module):
        def __init__(self, centroids, K, n_feat, eps=1e-7):
            super().__init__()
            self.K = K
            self.n_feat = n_feat
            self.eps = eps
            # FIXED: 'centroids' not 'centroits'
            self.register_buffer("centroids", centroids)

            self.log_tau = nn.Parameter(torch.tensor(-0.5))
            self.raw_w_d = nn.Parameter(torch.tensor(0.0))
            self.raw_w_m = nn.Parameter(torch.tensor(0.0))
            self.raw_w_e = nn.Parameter(torch.tensor(0.0))
            self.raw_w_s = nn.Parameter(torch.tensor(0.0))
            self.raw_feat_w = nn.Parameter(torch.zeros(n_feat))

        def _branch_weights(self):
            return torch.softmax(torch.stack([self.raw_w_d, self.raw_w_m,
                                              self.raw_w_e, self.raw_w_s]), dim=0)

        def _feat_weights(self):
            w = torch.softmax(self.raw_feat_w, dim=0)
            return w * self.n_feat

        @staticmethod
        def _phi_dir(x, eps):
            return torch.acos(torch.clamp(x, -1.0 + eps, 1.0 - eps))

        @staticmethod
        def _phi_mag(x, eps):
            return torch.acos(torch.clamp(x.abs(), 0.0, 1.0 - eps))

        @staticmethod
        def _euc(x, eps):
            return torch.tanh(torch.acos(torch.clamp(x, -1.0 + eps, 1.0 - eps)) * 0.8)

        @staticmethod
        def _shape(x, eps):
            mu = x.mean(1, keepdim=True)
            sd = x.std(1, keepdim=True).clamp(min=eps)
            return (x - mu) / sd

        def _combined_features(self, x):
            eps = self.eps
            wb = self._branch_weights()
            wf = self._feat_weights()
            sqrt_wf = wf.sqrt()

            phi_d = self._phi_dir(x, eps) * sqrt_wf
            phi_m = self._phi_mag(x, eps) * sqrt_wf
            euc   = self._euc(x, eps) * sqrt_wf
            shape = self._shape(x, eps)

            s_d = wb[0].sqrt()
            s_m = wb[1].sqrt()
            s_e = wb[2].sqrt()
            s_s = wb[3].sqrt()

            return torch.cat([s_d * phi_d, s_m * phi_m, s_e * euc, s_s * shape], dim=1)

        def _distance_matrix(self, x, y):
            fx = self._combined_features(x)
            fy = self._combined_features(y)
            sq_fx = (fx ** 2).sum(1, keepdim=True)
            sq_fy = (fy ** 2).sum(1, keepdim=True)
            dot = fx @ fy.T
            dist_sq = sq_fx + sq_fy.T - 2 * dot
            return torch.sqrt(torch.clamp(dist_sq, min=self.eps))

        def forward(self, x):
            tau = torch.exp(self.log_tau)
            d = self._distance_matrix(x, self.centroids)
            raw = 1.0 / (d / tau + self.eps)
            return raw / raw.sum(-1, keepdim=True)

    # ----------------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------------
    def _loss(self, probs, targets, class_weights):
        K = probs.shape[1]
        nll = nn.NLLLoss(weight=class_weights)(torch.log(probs.clamp(1e-10)), targets)
        s = self.label_smoothing / K
        ce = (1 - self.label_smoothing + s) * nll + s * (-torch.log(probs.clamp(1e-10)).mean())
        fw_reg = self.feat_weight_reg * ((self.net_._feat_weights() - 1.0)**2).mean()
        return ce + fw_reg

    # ----------------------------------------------------------------------
    # Fit
    # ----------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n_feat = X.shape[1]

        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([label_to_idx[c] for c in y], dtype=np.int64)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Train/val split
        n_val = max(K, int(len(X) * self.val_fraction))
        perm = rng.permutation(len(X))
        va_idx, tr_idx = perm[:n_val], perm[n_val:]
        X_tr, y_tr = X[tr_idx], y_enc[tr_idx]
        X_va, y_va = X[va_idx], y_enc[va_idx]

        self._norm_fit(X_tr)
        X_tr_norm = self._norm_apply(X_tr)
        X_va_norm = self._norm_apply(X_va)

        def tt(x, long=False):
            return torch.tensor(x, dtype=torch.long if long else torch.float32)

        # Class weights
        effective_cap = self.weight_cap
        cw = compute_class_weight('balanced', classes=self.classes_, y=y[tr_idx])
        cw_t = torch.tensor(cw, dtype=torch.float32).clamp(max=effective_cap)
        cw_t = cw_t / cw_t.sum() * K

        # Centroids
        centroids_np = np.stack([X_tr_norm[y_tr == i].mean(0) for i in range(K)])
        centroids = tt(centroids_np)

        # Build net
        self.net_ = self._Net(centroids, K, n_feat)

        opt = optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2, eta_min=1e-5)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tt(X_tr_norm), tt(y_tr, long=True)),
            batch_size=self.batch_size, shuffle=True)

        X_va_t = tt(X_va_norm)
        best_f1 = -1.0
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            self.net_.train()
            for xb, yb in loader:
                opt.zero_grad()
                probs = self.net_(xb)
                loss = self._loss(probs, yb, cw_t.to(xb.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            scheduler.step()

            self.net_.eval()
            with torch.no_grad():
                val_probs = self.net_(X_va_t).numpy()
            pred = val_probs.argmax(1)
            vf1 = f1_score(y_va, pred, average='macro', zero_division=0)

            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if self.verbose and (epoch+1) % 10 == 0:
                bw = self.net_._branch_weights().detach().numpy()
                print(f"epoch {epoch+1:3d} | val F1={vf1:.4f} | branches=[{bw[0]:.2f},{bw[1]:.2f},{bw[2]:.2f},{bw[3]:.2f}]")

            if wait >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.best_val_f1_ = best_f1

        # Extract parameters
        with torch.no_grad():
            self.branch_weights_ = self.net_._branch_weights().numpy()
            self.feat_weights_ = self.net_._feat_weights().numpy()
            self.tau_ = torch.exp(self.net_.log_tau).item()

        # Gravity weights
        n_tr = len(y_tr)
        counts = np.array([(y_tr == i).sum() for i in range(K)], dtype=np.float32)
        balanced = n_tr / (K * counts)
        balanced = balanced / balanced.mean()
        balanced = np.clip(balanced, 1.0/self.gravity_cap, self.gravity_cap)
        balanced = balanced / balanced.mean()
        self.gravity_weights_ = balanced.astype(np.float32)

        # Build combined vector function (for inference)
        def make_combined(x_norm):
            eps = 1e-7
            wb = self.branch_weights_
            wf = self.feat_weights_
            sqrt_wf = np.sqrt(wf)
            s_d = np.sqrt(wb[0]); s_m = np.sqrt(wb[1]); s_e = np.sqrt(wb[2]); s_s = np.sqrt(wb[3])

            phi_d = np.arccos(np.clip(x_norm, -1+eps, 1-eps)) * sqrt_wf
            phi_m = np.arccos(np.clip(np.abs(x_norm), 0, 1-eps)) * sqrt_wf
            euc   = np.tanh(np.arccos(np.clip(x_norm, -1+eps, 1-eps)) * 0.8) * sqrt_wf
            mu    = x_norm.mean(1, keepdims=True)
            sd    = x_norm.std(1, keepdims=True).clip(min=eps)
            shp   = (x_norm - mu) / sd
            return np.concatenate([s_d*phi_d, s_m*phi_m, s_e*euc, s_s*shp], axis=1).astype(np.float32)

        self._make_combined = make_combined

        # Centroids in combined space
        centroids_norm = self.net_.centroids.numpy()
        self.centroids_combined_ = make_combined(centroids_norm)

        # FAISS index for KNN
        self.lambda_ = self.lam_init
        if not _HAS_FAISS:
            if self.verbose:
                print("FAISS not installed, KNN disabled. λ forced to 1.0.")
            self.lambda_ = 1.0
            self.index_ = None
        else:
            X_train_combined = make_combined(X_tr_norm)
            dim = X_train_combined.shape[1]
            if self.use_ivf and X_train_combined.shape[0] >= 78:
                nlist = max(1, min(self.nlist, X_train_combined.shape[0] // 39))
                quantizer = faiss.IndexFlatL2(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
                index.train(X_train_combined)
                index.nprobe = self.nprobe
            else:
                index = faiss.IndexFlatL2(dim)
            index.add(X_train_combined)
            self.index_ = index
            self.train_class_idx_ = y_tr

            # Auto λ search
            if self.auto_lambda:
                X_val_combined = make_combined(X_va_norm)
                avm_probs = self._avm_proba(X_val_combined)
                knn_probs = self._knn_proba(X_val_combined)

                best_lam = self.lam_init
                best_f1 = -1.0
                for lam_cand in np.arange(0.0, 1.01, 0.1):
                    p = lam_cand * avm_probs + (1 - lam_cand) * knn_probs
                    if self.gravity != 0.0:
                        p = p * (self.gravity_weights_[None, :] ** self.gravity)
                        p = p / p.sum(1, keepdims=True)
                    f1 = f1_score(y_va, p.argmax(1), average='macro', zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_lam = lam_cand
                self.lambda_ = best_lam
                if self.verbose:
                    print(f"Lambda search: best λ = {best_lam:.2f} (val F1 = {best_f1:.4f})")

        return self

    # ----------------------------------------------------------------------
    # Probability helpers
    # ----------------------------------------------------------------------
    def _avm_proba(self, combined):
        a_sq = (combined ** 2).sum(1, keepdims=True)
        b_sq = (self.centroids_combined_ ** 2).sum(1)
        ab = combined @ self.centroids_combined_.T
        d = np.sqrt(np.maximum(a_sq + b_sq[None, :] - 2.0 * ab, 0.0))
        raw = 1.0 / (d + 1e-10)
        return raw / raw.sum(1, keepdims=True)

    def _knn_proba(self, combined):
        if self.index_ is None:
            raise RuntimeError("FAISS index not built.")
        dist, idx = self.index_.search(combined, self.k)
        w = 1.0 / (dist + 1e-10)
        w = w / w.sum(1, keepdims=True)
        n_test = combined.shape[0]
        K = len(self.classes_)
        proba = np.zeros((n_test, K), dtype=np.float32)
        t_idx = np.repeat(np.arange(n_test), self.k)
        np.add.at(proba, (t_idx, self.train_class_idx_[idx.ravel()]), w.ravel())
        return proba

    def predict_proba(self, X):
        check_is_fitted(self, 'centroids_combined_')
        X = np.asarray(X, dtype=np.float32)
        X_norm = self._norm_apply(X)
        combined = self._make_combined(X_norm)

        avm = self._avm_proba(combined)
        if self.lambda_ >= 1.0 or self.index_ is None:
            proba = avm
        else:
            knn = self._knn_proba(combined)
            proba = self.lambda_ * avm + (1 - self.lambda_) * knn

        if self.gravity != 0.0:
            proba = proba * (self.gravity_weights_[None, :] ** self.gravity)
            proba = proba / proba.sum(1, keepdims=True)
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def summary(self, feature_names=None):
        check_is_fitted(self, 'branch_weights_')
        # FIX: use 'is not None' instead of implicit truthiness
        if feature_names is not None:
            names = list(feature_names)
        else:
            names = [f"f{i}" for i in range(len(self.feat_weights_))]
        lines = [
            f"tau = {self.tau_:.3f}",
            f"lambda = {self.lambda_:.3f}  (auto_tuned={self.auto_lambda})",
            f"gravity = {self.gravity:.2f}  (cap={self.gravity_cap})",
            f"Branch weights: d={self.branch_weights_[0]:.3f}, m={self.branch_weights_[1]:.3f}, "
            f"e={self.branch_weights_[2]:.3f}, s={self.branch_weights_[3]:.3f}",
        ]
        order = np.argsort(self.feat_weights_)[::-1]
        lines.append("Top feature weights:")
        for i in order[:5]:
            lines.append(f"  {names[i]:<20} {self.feat_weights_[i]:.4f}")
        if self.gravity != 0.0:
            gw = self.gravity_weights_
            lines.append("Gravity weights per class:")
            for c, w in zip(self.classes_, gw):
                lines.append(f"  class {c}: {w:.3f}")
        return "\n".join(lines)
