Normalise each number to a value between -1 and +1 (so all features live in the same cube).

Six branches each turn the original numbers into a new set of numbers that highlight a different geometric property (e.g., “how close to +1”, “how far from the corners”, “the pattern across features”).

Combine all six branch outputs into one long vector per sample – this vector is the “fingerprint” of the sample.

Learn class prototypes (ideal fingerprints) plus branch weights, feature weights, and a “temperature” (sharpness).

Calculate how far the sample’s fingerprint is from each class prototype – convert distance to a score.

Convert scores to class probabilities.

Also compute KNN probabilities (looking at nearest neighbours in a separate “KNN‑space”).

Blend AVM and KNN probabilities using a learned weight (or a per‑sample confidence gate).

Train by minimising a loss function that includes:

Standard classification error,

Extra penalty for ordered classes (Earth Mover’s Distance),

Contrastive loss (pull same‑class points together),

Regularisation to keep prototypes stable and branches diverse.

After training, automatically choose the best blend (AVM vs KNN) on a validation set.

Predict by applying the same transformations to new data and using the blended probabilities.
