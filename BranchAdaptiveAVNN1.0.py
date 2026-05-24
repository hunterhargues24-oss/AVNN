"""
BranchAdaptiveAVNN
==================
Three‑branch learnable AVNN classifier.

Branches:
  - Angular: axis‑separable per‑feature arccos differences (weighted by learned feature weights)
  - Euclidean: L2 on tanh_ac‑transformed features
  - Shape: L2 between per‑sample z‑score normalised vectors

All three distances are blended using learnable softmax weights.
The model also learns α (angular vs Euclidean) ??? Wait, no – the blend weights replace α.
Actually, we remove the separate α and let the branch weights directly combine the three distances.
The KNN branch uses the same blended distance; AVM branch also uses the same blended distance.
Additional learnable parameters:
  - per‑feature angular weights (softmax, sum = n_features)
  - temperature τ for AVM scoring
  - λ for blending AVM and KNN
  - branch weights w_ang, w_euc, w_shape (softmax, sum=1)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import f1_score


class BranchAdaptiveAVNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5, lr=1e-3, epochs=300, batch_size=64,
                 patience=60, val_fraction=0.15, label_smoothing=0.05,
                 weight_cap=1.5, weight_decay=1e-4,
                 norm_range='[-1,1]', random_state=None, verbose=False):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.label_smoothing = label_smoothing
        self.weight_cap = weight_cap
        self.weight_decay = weight_decay
        self.norm_range = norm_range
        self.random_state = random_state
        self.verbose = verbose

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
        else:
            x = np.clip(x, 0.0, 1.0)
            return np.tanh(np.arccos(x) * 1.6).astype(np.float32)

    # ---------- inner PyTorch module ----------
    class _Net(nn.Module):
        def __init__(self, c_norm, c_trans, Xn, Xt, y_train, K, k, eps=1e-7):
            super().__init__()
            self.K, self.k, self.eps = K, k, eps
            n_feat = c_norm.shape[1]

            self.register_buffer("c_norm", c_norm)
            self.register_buffer("c_trans", c_trans)
            self.register_buffer("Xn", Xn)
            self.register_buffer("Xt", Xt)
            self.register_buffer("y_train", y_train)

            # Learnable parameters
            self.raw_lambda = nn.Parameter(torch.tensor(0.85))
            self.log_tau = nn.Parameter(torch.tensor(-0.5))
            self.raw_weights_ang = nn.Parameter(torch.zeros(n_feat))  # per-feature angular weights
            self.raw_w_ang = nn.Parameter(torch.tensor(0.0))   # branch weight for angular
            self.raw_w_euc = nn.Parameter(torch.tensor(0.0))   # branch weight for euclidean
            self.raw_w_shape = nn.Parameter(torch.tensor(0.0)) # branch weight for shape

        def _feat_weights(self):
            w = torch.softmax(self.raw_weights_ang, dim=0)
            return w * len(self.raw_weights_ang)

        def _branch_weights(self):
            return torch.softmax(torch.stack([self.raw_w_ang, self.raw_w_euc, self.raw_w_shape]), dim=0)

        # ---------- distance components ----------
        def _angular_per_feat(self, xn_a, xn_b):
            lo = -1.0 + self.eps
            hi = 1.0 - self.eps
            th_a = torch.acos(torch.clamp(xn_a, lo, hi))
            th_b = torch.acos(torch.clamp(xn_b, lo, hi))
            diff = torch.abs(th_a.unsqueeze(1) - th_b.unsqueeze(0))
            return diff   # (N, M, F)

        def _angular_dist(self, xn_a, xn_b):
            w = self._feat_weights()
            diff = self._angular_per_feat(xn_a, xn_b)  # (N, M, F)
            ang = (diff * w).sum(dim=-1)               # (N, M)
            return ang

        def _euclidean_dist(self, xt_a, xt_b):
            diff = xt_a.unsqueeze(1) - xt_b.unsqueeze(0)
            return torch.sqrt((diff**2).sum(-1) + self.eps)

        def _shape_dist(self, xn_a, xn_b):
            eps = self.eps
            mu_a = xn_a.mean(dim=1, keepdim=True)
            sd_a = xn_a.std(dim=1, keepdim=True).clamp(min=eps)
            mu_b = xn_b.mean(dim=1, keepdim=True)
            sd_b = xn_b.std(dim=1, keepdim=True).clamp(min=eps)
            za = (xn_a - mu_a) / sd_a
            zb = (xn_b - mu_b) / sd_b
            diff = za.unsqueeze(1) - zb.unsqueeze(0)
            return torch.sqrt((diff**2).sum(-1) + eps)

        def _blended_dist(self, xn_a, xt_a, xn_b, xt_b):
            w_branch = self._branch_weights()
            d_ang = self._angular_dist(xn_a, xn_b)
            d_euc = self._euclidean_dist(xt_a, xt_b)
            d_shape = self._shape_dist(xn_a, xn_b)
            return w_branch[0] * d_ang + w_branch[1] * d_euc + w_branch[2] * d_shape

        def _avm_proba(self, xn, xt):
            tau = torch.exp(self.log_tau)
            d = self._blended_dist(xn, xt, self.c_norm, self.c_trans)
            raw = 1.0 / (d / tau + self.eps)
            return raw / raw.sum(dim=-1, keepdim=True)

        def _knn_proba(self, xn, xt):
            d = self._blended_dist(xn, xt, self.Xn, self.Xt)
            k_eff = min(self.k, self.Xn.shape[0])
            topk = torch.topk(d, k=k_eff, dim=1, largest=False)
            w = 1.0 / (topk.values + self.eps)
            w = w / w.sum(dim=1, keepdim=True)
            y_neigh = self.y_train[topk.indices]
            y_onehot = F.one_hot(y_neigh, num_classes=self.K).float()
            return (w.unsqueeze(-1) * y_onehot).sum(dim=1)

        def forward(self, xn, xt):
            lam = torch.sigmoid(self.raw_lambda)
            return lam * self._avm_proba(xn, xt) + (1 - lam) * self._knn_proba(xn, xt)

    # ---------- loss ----------
    def _loss(self, probs, targets, class_w):
        K = probs.shape[1]
        nll = nn.NLLLoss(weight=class_w)(torch.log(probs.clamp(1e-10)), targets)
        s = self.label_smoothing / K
        return (1 - self.label_smoothing + s) * nll + s * (-torch.log(probs.clamp(1e-10)).mean())

    # ---------- fit ----------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([label_to_idx[c] for c in y], dtype=np.int64)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        n_val = max(K, int(len(X) * self.val_fraction))
        perm = rng.permutation(len(X))
        va_idx, tr_idx = perm[:n_val], perm[n_val:]
        X_tr, y_tr = X[tr_idx], y_enc[tr_idx]
        X_va, y_va = X[va_idx], y_enc[va_idx]

        self._norm_fit(X_tr)
        Xtrn = self._norm_apply(X_tr)
        Xtrx = self._tanh_ac(Xtrn)
        Xvan = self._norm_apply(X_va)
        Xvax = self._tanh_ac(Xvan)

        def tt(a, long=False):
            return torch.tensor(a, dtype=torch.long if long else torch.float32)

        cw = compute_class_weight('balanced', classes=self.classes_, y=y[tr_idx])
        cw_t = torch.tensor(cw, dtype=torch.float32).clamp(max=self.weight_cap)
        cw_t = cw_t / cw_t.sum() * K

        c_norm = tt(np.stack([Xtrn[y_tr == i].mean(0) for i in range(K)]))
        c_trans = tt(np.stack([Xtrx[y_tr == i].mean(0) for i in range(K)]))

        Xtrn_t = tt(Xtrn); Xtrx_t = tt(Xtrx); ytr_t = tt(y_tr, long=True)
        Xvan_t = tt(Xvan); Xvax_t = tt(Xvax)

        self.net_ = self._Net(c_norm, c_trans, Xtrn_t, Xtrx_t, ytr_t, K, self.k)

        opt = optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2, eta_min=1e-5)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtrn_t, Xtrx_t, ytr_t),
            batch_size=self.batch_size, shuffle=True)

        best_f1, best_state, wait = -1.0, None, 0
        for epoch in range(self.epochs):
            self.net_.train()
            for xn, xt, yb in loader:
                opt.zero_grad()
                self._loss(self.net_(xn, xt), yb, cw_t).backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            sch.step()

            self.net_.eval()
            with torch.no_grad():
                pred_idx = self.net_(Xvan_t, Xvax_t).argmax(-1).numpy()
            vf1 = f1_score(y_va, pred_idx, average='macro', zero_division=0)

            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                bw = self.net_._branch_weights().detach().numpy()
                lam = torch.sigmoid(self.net_.raw_lambda).item()
                tau = torch.exp(self.net_.log_tau).item()
                w = torch.softmax(self.net_.raw_weights_ang, dim=0).detach().numpy()
                print(f"  epoch {epoch+1:3d}  val_f1={vf1:.4f}  lr={opt.param_groups[0]['lr']:.5f}")
                print(f"    branch_weights: ang={bw[0]:.3f} euc={bw[1]:.3f} shape={bw[2]:.3f}")
                print(f"    lam={lam:.3f} tau={tau:.3f}")
                print(f"    angular_feat_weights={np.round(w, 3)}")

            if wait >= self.patience:
                if self.verbose:
                    print(f"  early stopping at epoch {epoch+1}")
                break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.best_val_f1_ = best_f1

        with torch.no_grad():
            self.lambda_ = torch.sigmoid(self.net_.raw_lambda).item()
            self.tau_ = torch.exp(self.net_.log_tau).item()
            self.angular_feat_weights_ = torch.softmax(self.net_.raw_weights_ang, dim=0).numpy()
            self.branch_weights_ = self.net_._branch_weights().numpy()

        return self

    # ---------- predict ----------
    def _prepare(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._norm_apply(X)
        Xt = self._tanh_ac(Xn)
        return torch.tensor(Xn), torch.tensor(Xt)

    def predict_proba(self, X):
        check_is_fitted(self, 'net_')
        xn, xt = self._prepare(X)
        with torch.no_grad():
            return self.net_(xn, xt).numpy()

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def summary(self, feature_names=None):
        names = feature_names or [f"f{i}" for i in range(len(self.angular_feat_weights_))]
        lines = [f"lambda={self.lambda_:.4f}, tau={self.tau_:.4f}"]
        lines.append(f"Branch weights: angular={self.branch_weights_[0]:.4f}, "
                     f"euclidean={self.branch_weights_[1]:.4f}, shape={self.branch_weights_[2]:.4f}")
        lines.append("Angular per‑feature weights:")
        for n, w in zip(names, self.angular_feat_weights_):
            lines.append(f"  {n:<24} {w:.4f}")
        return "\n".join(lines)
