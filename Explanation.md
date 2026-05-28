What the model is trying to do

Every classifier asks the same question: given a new sample, which class does it most likely belong to? Most models draw lines or curves between classes. This model instead asks: how similar is this sample to each class, from multiple geometric perspectives simultaneously? The answer is a weighted vote across three independent geometric views.

Step 1 — Normalisation

Every feature gets scaled to fit between -1 and +1. This puts all features on the same playing field so that alcohol content and pH are measured in the same geometric space.

Step 2 — The six branches

Before measuring any distance, the model transforms each sample through six different lenses. Think of these as six photographers taking pictures of the same room from different angles — each captures real information the others miss.

The AVM branches (used for measuring distance to class prototypes):

tanh_arccos — measures how far a feature is from the centre of its range. A feature at +0.9 or -0.9 gets a very different reading than a feature at +0.1. It’s most sensitive in the middle of the range, where subtle differences between classes often live. This is the dominant branch on almost every dataset.

boundary — the only branch that looks at all features simultaneously. It asks “where is this point relative to the hypercube as a whole?” — capturing cross-feature relationships that no single-feature branch can see.

circular — asks “how extreme is this feature, regardless of direction?” A value of +0.9 and -0.9 look the same to it. It catches when a feature is near either limit without caring which limit.

The KNN branches (used for finding nearest neighbours):

linear — raw signed position. Feature 3 is at value 0.7. That’s it.

shape — ignores absolute values entirely. It asks “within this sample, which features are relatively high and which are relatively low?” Two wines with completely different absolute values but the same relative profile look identical to this branch.

quadratic — how far from zero, with nonlinear emphasis near the extremes. Similar to circular but with different curvature.

The branch weights are learned — the model discovers which perspectives matter most for a given dataset. On ArrivalType, tanh_arccos earns nearly all the weight. On wine cultivar, shape earns more weight because cultivar identity is about feature profiles.

Step 3 — AVM: distance to class prototypes

The three AVM branches produce a combined vector for each sample. The model has learned a prototype for each class — a point in that combined space representing the “typical” member of that class. The AVM score for each class is how close the sample is to that class’s prototype, measured via inverse distance with a learned Mahalanobis covariance that accounts for the actual shape and orientation of each class cluster.

Close to a prototype → high score → high probability for that class.

The triangle scoring folds into AVM. For every pair of features, it asks whether the sample’s relative feature ratios match the prototype’s. A wine that is close to the high-quality centroid in absolute distance but has the wrong alcohol-to-acidity ratio gets penalised by triangle, pulling its AVM score down.

AVM answers: does this sample look like it belongs to class k, from a global perspective?

Step 4 — KNN: who are the actual neighbours?

Completely independent of prototypes. The three KNN branches produce a separate combined vector. The model uses FAISS to find the k most similar training samples in that space and takes a weighted vote — samples that are closer vote louder.

KNN answers: what classes do the actual nearby training samples belong to?

KNN and AVM often agree. Where they disagree is informative — AVM might be confused near a decision boundary where two class prototypes are equidistant, but KNN sees a clear local majority and is confident. Or KNN might be confused in a sparse region where the nearest neighbours are far away, but AVM knows this point is geometrically consistent with a class centroid.

Step 5 — The lambda gate

The final prediction blends AVM and KNN:

final = λ · AVM + (1 - λ) · KNN


Lambda is determined one of two ways after training, whichever scores better on the held-out validation set:

Scalar lambda — a single fixed number (e.g. 0.80) applied to every prediction. Chosen by grid search. Means “trust AVM 80% and KNN 20% for every sample.”

Confidence gate — per-sample lambda based on how confident each component is. Confidence is measured by entropy — a flat probability distribution across classes means uncertain, a peaked distribution means confident. If AVM is confident and KNN isn’t, lambda goes high. If KNN is confident and AVM isn’t, lambda goes low. If both are confident but disagree — a standoff — lambda sits near 0.5 and the disagreement score flags the prediction for human review.

The model picks whichever approach scored better on validation. Lambda: 0.800 means scalar won. Lambda: [confidence gate] means per-sample won.

The full flow in one sentence per step

	1.	Scale features to [-1, +1]
	2.	Transform through six branch lenses, split into AVM space and KNN space
	3.	AVM measures how geometrically consistent the sample is with each class prototype, folding in triangle’s profile-matching signal
	4.	KNN counts what class the actual nearby training samples are
	5.	The lambda gate decides how much to trust each answer, either globally or per-sample based on confidence

Why it works well on imbalanced data

Rare classes fail different checks on different samples. Some minority samples are close to the centroid but have the wrong feature profile — triangle catches that. Some have the right profile but are in a sparse neighbourhood — AVM catches that. Some are near the majority boundary where KNN would be fooled by majority neighbours — AVM’s centroid geometry holds firm. No single component catches every failure mode. The gate lets whichever component is most confident lead.
