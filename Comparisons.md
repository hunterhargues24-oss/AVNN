The most important number is the large dataset fit time. AutoLambdaAVNN takes ~7 seconds on 200k samples — faster than XGBoost — because training is O(K) not O(N). Every epoch runs against K=4 centroids, not 200,000 training points. XGBoost builds trees over all N samples, which is why it slows down at scale. The FAISS index build is the only step that touches all N.

**Where XGBoost clearly wins:** small balanced datasets, raw accuracy, and fit speed on small data. On red wine at 1,599 samples, XGBoost fits in under a second. AutoLambdaAVNN takes 8-15 seconds for 90-150 epochs. That gap is real and worth knowing.

**Where AutoLambdaAVNN wins:** macro F1 on imbalanced tabular data. Every imbalanced dataset in the benchmark — red wine, white wine, ArrivalType — shows AVNN variants outperforming XGBoost on macro F1. XGBoost needs SMOTE or `scale_pos_weight` tuning to compete; AVNN handles it geometrically with gravity and ordinal loss built in.

**vs deep learning:** This is not a fair fight in AVNN's favor on large high-dimensional data. A TabNet or properly tuned MLP on 200k samples with GPU would likely beat AVNN on accuracy. The advantage is interpretability — `volatile_acidity=2.23, alcohol=1.94` means something concrete. Neural network weights do not.

**The honest positioning:** AVNN is a strong alternative to XGBoost specifically for imbalanced tabular classification where you need to detect rare events and care about macro F1. That describes your ArrivalType use case exactly.
