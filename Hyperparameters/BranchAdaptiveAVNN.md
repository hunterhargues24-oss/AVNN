2. Learnable BranchAdaptiveAVNN (PyTorch)
Inherits static hyperparameters plus training‑related ones and learnable parameter initialisations.

Hyperparameter	Default	Description
k	5	Number of neighbours (used at inference only).
lr	1e-3	Learning rate for Adam (applies to α, λ, τ, branch weights, feature weights).
epochs	300	Maximum training epochs.
batch_size	64	Mini‑batch size.
patience	60	Early stopping patience (validation macro F1).
val_fraction	0.15	Fraction of training data held out for validation (stratified).
label_smoothing	0.05	Label smoothing factor for NLL loss.
weight_cap	1.5	Maximum class weight (sklearn balanced weights capped, then renormalised).
weight_decay	1e-4	Adam weight decay (L2 regularisation).
norm_range	'[-1,1]'	Same as static version.
random_state	None	Seed for reproducibility.
verbose	False	Print training progress.
Learnable parameters (initial values):

alpha initialised via raw_alpha = 0.0 (sigmoid → 0.5)

lambda initialised via raw_lambda = 0.85 (sigmoid → ~0.70)

tau initialised via log_tau = -0.5 (exp → ~0.61)

Branch weights (w_ang, w_euc, w_shape) initialised equal (softmax of zeros → each ≈0.333)

Per‑feature angular weights initialised uniform (softmax → equal, scaled to sum = number of features)
