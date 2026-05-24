1. Static AVNNClassifier (parameter‑free, no training)
Hyperparameter	Default	Description
alpha	0.5	Blend between angular and Euclidean distance (0 = pure Euclidean, 1 = pure angular).
lam	0.7	Blend between AVM centroid branch (1) and KNN branch (0).
k	5	Number of nearest neighbours for KNN branch.
transform	'tanh_ac'	Transform applied after normalisation for Euclidean branch. Options: 'tanh_ac', 'identity', 'tan', 'neglog'.
norm_range	'[-1,1]'	Normalisation range. Options: '[-1,1]' (full semicircle) or '[0,1]' (quarter‑circle).
eps	1e‑10	Small constant to avoid division by zero.
No other parameters – centroids are computed as class means.
