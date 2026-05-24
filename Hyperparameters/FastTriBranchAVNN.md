3. FAISS‑Accelerated FastTriBranchAVNN (static, no training)
Hyperparameter	Default	Description
lam	0.7	Same blend between AVM and KNN.
k	10	Number of nearest neighbours (larger than static because FAISS is fast).
w_ang, w_euc, w_shape	0.34, 0.33, 0.33	Branch weights (can be set manually or from a trained model).
feat_weights	None	Per‑feature angular weights (if None, uniform).
transform	'tanh_ac'	Same as static.
norm_range	'[-1,1]'	Same.
eps	1e-10	Same.
use_ivf	True	Whether to use IVF index (approximate).
nlist	1000	Number of Voronoi cells for IVF.
nprobe	10	Number of cells to search (higher = slower but more accurate).
Note: This model does not train. It uses the provided branch weights (e.g., from a trained BranchAdaptiveAVNN) or default
