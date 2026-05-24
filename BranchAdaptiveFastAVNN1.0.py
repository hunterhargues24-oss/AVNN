"""
BranchAdaptiveFastAVNN – Learnable branch weights + FAISS KNN (non‑trainable).
- Training: pure AVM (lam=1.0) with PyTorch on full training set (batch‑wise).
- Inference: uses FAISS for fast KNN with learned branch weights.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import faiss
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import f1_score

class BranchAdaptiveFastAVNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5, lr=1e-3, epochs=300, batch_size=64,
                 patience=60, val_fraction=0.15, label_smoothing=0.05,
                 weight_cap=1.5, weight_decay=1e-4,
                 norm_range='[-1,1]', random_state=None, verbose=False,
                 use_ivf=True, nlist=1000, nprobe=10):
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
        self.use_ivf = use_ivf
        self.nlist = nlist
        self.nprobe = nprobe

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

    def _get_transform_fn(self):
        if callable(self.transform):
            return self.transform
        return {
            'tanh_ac': self._tanh_ac,
            'identity': lambda x: x,
            'tan': lambda x: np.tan(np.clip(x, 0, 1) * np.pi / 2 * 0.97),
            'neglog': lambda x: -np.log(np.clip(x, 1e-10, 1.0)),
        }[self.transform]

    # ---------- PyTorch model (pure AVM) ----------
    class _Net(nn.Module):
        def __init__(self, c_norm, c_trans, K, n_feat, eps=1e-7):
            super().__init__()
            self.K = K
            self.eps = eps
            self.register_buffer("c_norm", c_norm)
            self.register_buffer("c_trans", c_trans)
            self.log_tau = nn.Parameter(torch.tensor(-0.5))
            self.raw_weights_ang = nn.Parameter(torch.zeros(n_feat))
            self.raw_w_ang = nn.Parameter(torch.tensor(0.0))
            self.raw_w_euc = nn.Parameter(torch.tensor(0.0))
            self.raw_w_shape = nn.Parameter(torch.tensor(0.0))
            self.raw_lambda = nn.Parameter(torch.tensor(0.85))  # not used during training but kept

        def _feat_weights(self):
            w = torch.softmax(self.raw_weights_ang, dim=0)
            return w * len(self.raw_weights_ang)

        def _branch_weights(self):
            return torch.softmax(torch.stack([self.raw_w_ang, self.raw_w_euc, self.raw_w_shape]), dim=0)

        def _angular(self, xn):
            lo = -1.0 + self.eps
            hi = 1.0 - self.eps
            return torch.acos(torch.clamp(xn, lo, hi))

        def _shape(self, xn):
            mu = xn.mean(dim=1, keepdim=True)
            sd = xn.std(dim=1, keepdim=True).clamp(min=self.eps)
            return (xn - mu) / sd

        def _blended_dist(self, xn_a, xt_a, xn_b, xt_b):
            w_branch = self._branch_weights()
            w_feat = self._feat_weights()
            # Angular
            ang_a = self._angular(xn_a)
            ang_b = self._angular(xn_b)
            ang = (torch.abs(ang_a.unsqueeze(1) - ang_b.unsqueeze(0)) * w_feat).sum(-1)
            # Euclidean
            diff = xt_a.unsqueeze(1) - xt_b.unsqueeze(0)
            euc = torch.sqrt((diff**2).sum(-1) + self.eps)
            # Shape
            shape_a = self._shape(xn_a)
            shape_b = self._shape(xn_b)
            shape = torch.sqrt(((shape_a.unsqueeze(1) - shape_b.unsqueeze(0))**2).sum(-1) + self.eps)
            return w_branch[0] * ang + w_branch[1] * euc + w_branch[2] * shape

        def _avm_proba(self, xn, xt):
            tau = torch.exp(self.log_tau)
            d = self._blended_dist(xn, xt, self.c_norm, self.c_trans)
            raw = 1.0 / (d / tau + self.eps)
            return raw / raw.sum(dim=-1, keepdim=True)

        def forward(self, xn, xt):
            # pure AVM (lambda=1.0) – no KNN during training
            return self._avm_proba(xn, xt)

    # ---------- train (pure AVM) ----------
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

        # Split train/val
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

        # Model
        n_feat = c_norm.shape[1]
        self.net_ = self._Net(c_norm, c_trans, K, n_feat)
        opt = optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2, eta_min=1e-5)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tt(Xtrn), tt(Xtrx), tt(y_tr, long=True)),
            batch_size=self.batch_size, shuffle=True
        )
        Xvan_t = tt(Xvan); Xvax_t = tt(Xvax); yva_t = tt(y_va, long=True)

        best_f1, best_state, wait = -1.0, None, 0
        for epoch in range(self.epochs):
            self.net_.train()
            total_loss = 0.0
            for xn, xt, yb in loader:
                opt.zero_grad()
                probs = self.net_(xn, xt)
                nll = nn.NLLLoss(weight=cw_t)(torch.log(probs.clamp(1e-10)), yb)
                s = self.label_smoothing / K
                loss = (1 - self.label_smoothing + s) * nll + s * (-torch.log(probs.clamp(1e-10)).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
                total_loss += loss.item() * len(xn)
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
                print(f"  epoch {epoch+1:3d} val_f1={vf1:.4f}")
            if wait >= self.patience:
                break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.best_val_f1_ = best_f1

        # Store learned parameters for inference
        with torch.no_grad():
            self.branch_weights_ = self.net_._branch_weights().numpy()
            self.angular_feat_weights_ = self.net_._feat_weights().numpy()
            self.tau_ = torch.exp(self.net_.log_tau).item()
            # lambda is not used during training but we keep it as default 0.7
            self.lambda_ = 0.7

        # Build FAISS index on full training set (using learned branch weights)
        self.transform = 'tanh_ac'  # same as in preproc
        X_all_norm = self._norm_apply(X_tr)   # use all training data (not just training subset? We have X_tr from split)
        X_all_trans = self._tanh_ac(X_all_norm)
        # Combine components using learned weights
        ang_all = np.arccos(np.clip(X_all_norm, -1.0 + 1e-7, 1.0 - 1e-7))
        euc_all = X_all_trans
        shape_all = (X_all_norm - X_all_norm.mean(axis=1, keepdims=True)) / (X_all_norm.std(axis=1, keepdims=True).clip(1e-7))
        w_ang, w_euc, w_shape = self.branch_weights_
        scale_ang = np.sqrt(w_ang)
        scale_euc = np.sqrt(w_euc)
        scale_shape = np.sqrt(w_shape)
        combined = np.concatenate([scale_ang * ang_all, scale_euc * euc_all, scale_shape * shape_all], axis=1)
        # Build FAISS index
        if self.use_ivf:
            nlist = min(self.nlist, combined.shape[0] // 39)
            if nlist < 1: nlist = 1
            quantizer = faiss.IndexFlatL2(combined.shape[1])
            index = faiss.IndexIVFFlat(quantizer, combined.shape[1], nlist, faiss.METRIC_L2)
            index.train(combined)
            index.nprobe = self.nprobe
        else:
            index = faiss.IndexFlatL2(combined.shape[1])
        index.add(combined)
        self.index_ = index
        self.y_train_ = y_tr
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        self.train_class_idx_ = np.array([self.class_to_idx_[c] for c in y_tr])

        return self

    # ---------- prediction (AVM + FAISS KNN) ----------
    def _prepare(self, X):
        X_norm = self._norm_apply(X)
        X_trans = self._tanh_ac(X_norm)
        return X_norm, X_trans

    def predict_proba(self, X):
        check_is_fitted(self, 'index_')
        X_norm, X_trans = self._prepare(X)

        # AVM branch using trained parameters
        # Recreate centroids in combined space? Actually we need centroids in the same combined space.
        # We have centroids stored? Not directly. We'll compute centroids on the fly from training data? Simpler: reuse the trained net's centroids.
        # But the net expects normalised inputs. We'll reuse the net for AVM branch.
        # Convert test inputs to tensors and run net (pure AVM)
        xn_t = torch.tensor(X_norm, dtype=torch.float32)
        xt_t = torch.tensor(X_trans, dtype=torch.float32)
        with torch.no_grad():
            avm_proba = self.net_(xn_t, xt_t).numpy()

        # FAISS KNN branch
        ang = np.arccos(np.clip(X_norm, -1.0 + 1e-7, 1.0 - 1e-7))
        euc = X_trans
        shape = (X_norm - X_norm.mean(axis=1, keepdims=True)) / (X_norm.std(axis=1, keepdims=True).clip(1e-7))
        w_ang, w_euc, w_shape = self.branch_weights_
        scale_ang = np.sqrt(w_ang)
        scale_euc = np.sqrt(w_euc)
        scale_shape = np.sqrt(w_shape)
        combined = np.concatenate([scale_ang * ang, scale_euc * euc, scale_shape * shape], axis=1)
        distances, indices = self.index_.search(combined, self.k)
        weights = 1.0 / (distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)
        n_test = combined.shape[0]
        n_classes = len(self.classes_)
        knn_proba = np.zeros((n_test, n_classes), dtype=np.float32)
        flat_indices = indices.ravel()
        flat_weights = weights.ravel()
        test_indices = np.repeat(np.arange(n_test), self.k)
        np.add.at(knn_proba, (test_indices, self.train_class_idx_[flat_indices]), flat_weights)

        proba = self.lambda_ * avm_proba + (1 - self.lambda_) * knn_proba
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def summary(self, feature_names=None):
        names = feature_names or [f"f{i}" for i in range(len(self.angular_feat_weights_))]
        lines = [f"lambda={self.lambda_:.4f}, tau={self.tau_:.4f}"]
        lines.append(f"Branch weights: angular={self.branch_weights_[0]:.4f}, "
                     f"euclidean={self.branch_weights_[1]:.4f}, shape={self.branch_weights_[2]:.4f}")
        lines.append("Angular per‑feature weights:")
        for n, w in zip(names, self.angular_feat_weights_):
            lines.append(f"  {n:<24} {w:.4f}")
        return "\n".join(lines)
