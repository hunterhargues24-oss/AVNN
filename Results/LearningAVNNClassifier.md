============================================================
Iris  (150 samples, 4 features, 3 classes)
============================================================
  Accuracy    : 0.9533 ± 0.0163
  Macro F1    : 0.9531 ± 0.0165
  Weighted F1 : 0.9531 ± 0.0165

Model summary (last fold):
  Classes (3): [np.int64(0), np.int64(1), np.int64(2)]
  Train/val split: 102 / 18

  Lambda : [confidence gate]
  Tau    : cls0:0.604 cls1:0.604 cls2:0.610
  Val F1 : 0.9442

  Mahalanobis : full RDA  (alpha=0.5  ridge=1e-06)
  Ordinal EMD : False  (weight=0.5)
  SupCon      : True  (weight=0.3  temp=0.1)
  EMA         : False  (beta=0.9)
  Triangle    : False  (weight=0.3)
  Dual bnd    : False
  n_prototypes: 1
  per_class_τ : True
  ortho_reg   : 0.0

  AVM branches (global structure):
    tanh_arccos : 0.3563
    boundary    : 0.3563
    circular    : 0.2874
  KNN branches (local structure) [gradient frozen]:
    linear      : 0.3333
    shape       : 0.3333
    quadratic   : 0.3333

  Feature weights (top 5):
    petal width (cm)             1.0049
    petal length (cm)            1.0049
    sepal width (cm)             0.9951
    sepal length (cm)            0.9951
  Feature weights (bottom 2):
    sepal length (cm)            0.9951
    sepal width (cm)             0.9951

  Centroid drift:
    class0: 0.0098  
    class1: 0.0098  
    class2: 0.0098  

============================================================
Wine  (178 samples, 13 features, 3 classes)
============================================================
  Accuracy    : 0.9832 ± 0.0223
  Macro F1    : 0.9831 ± 0.0222
  Weighted F1 : 0.9830 ± 0.0226

Model summary (last fold):
  Classes (3): [np.int64(0), np.int64(1), np.int64(2)]
  Train/val split: 123 / 20

  Lambda : 0.800
  Tau    : cls0:0.631 cls1:0.596 cls2:0.583
  Val F1 : 0.7029

  Mahalanobis : full RDA  (alpha=0.5  ridge=1e-06)
  Ordinal EMD : False  (weight=0.5)
  SupCon      : True  (weight=0.3  temp=0.1)
  EMA         : False  (beta=0.9)
  Triangle    : False  (weight=0.3)
  Dual bnd    : False
  n_prototypes: 1
  per_class_τ : True
  ortho_reg   : 0.0

  AVM branches (global structure):
    tanh_arccos : 0.3594
    boundary    : 0.3499
    circular    : 0.2907
  KNN branches (local structure) [gradient frozen]:
    linear      : 0.3333
    shape       : 0.3333
    quadratic   : 0.3333

  Feature weights (top 5):
    flavanoids                   1.0546
    color_intensity              1.0534
    od280/od315_of_diluted_wines 1.0452
    proline                      1.0407
    alcohol                      1.0276
  Feature weights (bottom 2):
    hue                          0.9656
    nonflavanoid_phenols         0.9659

  Centroid drift:
    class0: 0.1390  █████
    class1: 0.1592  ██████
    class2: 0.1566  ██████

  Lambda search: floor=0.5  chosen=0.800

============================================================
Breast Cancer  (569 samples, 30 features, 2 classes)
============================================================
  Accuracy    : 0.9491 ± 0.0116
  Macro F1    : 0.9448 ± 0.0125
  Weighted F1 : 0.9487 ± 0.0117

Model summary (last fold):
  Classes (2): [np.int64(0), np.int64(1)]
  Train/val split: 389 / 67

  Lambda : 0.750
  Tau    : cls0:0.621 cls1:0.592
  Val F1 : 0.8454

  Mahalanobis : full RDA  (alpha=0.5  ridge=1e-06)
  Ordinal EMD : True  (weight=0.5)
  SupCon      : True  (weight=0.3  temp=0.1)
  EMA         : False  (beta=0.9)
  Triangle    : False  (weight=0.3)
  Dual bnd    : False
  n_prototypes: 1
  per_class_τ : True
  ortho_reg   : 0.0

  AVM branches (global structure):
    tanh_arccos : 0.3585
    boundary    : 0.3525
    circular    : 0.2890
  KNN branches (local structure) [gradient frozen]:
    linear      : 0.3333
    shape       : 0.3333
    quadratic   : 0.3333

  Feature weights (top 5):
    area error                   1.0260
    mean area                    1.0249
    worst area                   1.0248
    mean concave points          1.0232
    perimeter error              1.0229
  Feature weights (bottom 2):
    texture error                0.9753
    worst concave points         0.9756

  Centroid drift:
    class0: 0.1301  █████
    class1: 0.1336  █████

  Lambda search: floor=0.5  chosen=0.750

============================================================
Red Wine Quality  (1599 samples, 11 features, 3 classes)
============================================================
  Accuracy    : 0.8255 ± 0.0405
  Macro F1    : 0.5904 ± 0.0346
  Weighted F1 : 0.8288 ± 0.0298

Model summary (last fold):
  Classes (3): [np.int64(0), np.int64(1), np.int64(2)]
  Train/val split: 1090 / 190

  Lambda : 0.800
  Tau    : cls0:0.578 cls1:0.636 cls2:0.632
  Val F1 : 0.5776

  Mahalanobis : full RDA  (alpha=0.5  ridge=1e-06)
  Ordinal EMD : True  (weight=0.5)
  SupCon      : True  (weight=0.3  temp=0.1)
  EMA         : True  (beta=0.9)
  Triangle    : True  (weight=0.2)
  Dual bnd    : False
  n_prototypes: 1
  per_class_τ : True
  ortho_reg   : 0.0

  AVM branches (global structure):
    tanh_arccos : 0.3595
    boundary    : 0.3505
    circular    : 0.2900
  KNN branches (local structure) [gradient frozen]:
    linear      : 0.3333
    shape       : 0.3333
    quadratic   : 0.3333

  Feature weights (top 5):
    volatile acidity             1.0423
    sulphates                    1.0353
    alcohol                      1.0248
    chlorides                    1.0070
    total sulfur dioxide         1.0034
  Feature weights (bottom 2):
    free sulfur dioxide          0.9667
    citric acid                  0.9699

  Centroid drift:
    class0: 0.0650  ██
    class1: 0.0571  ██
    class2: 0.0559  ██

  Lambda search: floor=0.5  chosen=0.800

============================================================
White Wine Quality  (4898 samples, 11 features, 3 classes)
============================================================
  Accuracy    : 0.6925 ± 0.0088
  Macro F1    : 0.6925 ± 0.0084
  Weighted F1 : 0.6926 ± 0.0087

Model summary (last fold):
  Classes (3): [np.int64(0), np.int64(1), np.int64(2)]
  Train/val split: 3333 / 586

  Lambda : 0.900
  Tau    : cls0:0.639 cls1:0.563 cls2:0.640
  Val F1 : 0.3265

  Mahalanobis : full RDA  (alpha=0.5  ridge=1e-06)
  Ordinal EMD : True  (weight=0.5)
  SupCon      : True  (weight=0.3  temp=0.1)
  EMA         : True  (beta=0.9)
  Triangle    : True  (weight=0.1)
  Dual bnd    : False
  n_prototypes: 1
  per_class_τ : True
  ortho_reg   : 0.0

  AVM branches (global structure):
    tanh_arccos : 0.3576
    boundary    : 0.3487
    circular    : 0.2937
  KNN branches (local structure) [gradient frozen]:
    linear      : 0.3333
    shape       : 0.3333
    quadratic   : 0.3333

  Feature weights (top 5):
    density                      1.0412
    alcohol                      1.0351
    chlorides                    1.0337
    volatile acidity             1.0162
    free sulfur dioxide          0.9944
  Feature weights (bottom 2):
    residual sugar               0.9595
    sulphates                    0.9726

  Centroid drift:
    class0: 0.0906  ███
    class1: 0.0471  █
    class2: 0.0893  ███

  Lambda search: floor=0.5  chosen=0.900

ArrivalType.csv not found – skipping large-dataset test.
