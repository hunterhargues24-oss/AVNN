============================================================
Iris  (150 samples, 4 features, 3 classes)
============================================================
  Accuracy    : 0.9600 ± 0.0389
  Macro F1    : 0.9598 ± 0.0390
  Weighted F1 : 0.9598 ± 0.0390

Model summary (last fold):
  Classes (3): 0: 33%  1: 33%  2: 33%
  Train / val  : 102 / 18

  Lambda  : [confidence gate]
  Tau     : cls0:0.604  cls1:0.604  cls2:0.610
  Val F1  : 0.9442

  Mahalanobis : full RDA  (alpha=0.5  ridge=1e-06)
  Ordinal EMD : False  (weight=0.5)
  SupCon      : True  (weight=0.3  temp=0.1)
  EMA         : False  (beta=0.9)
  Dual bnd    : False
  Boundary init: True
  n_prototypes: 1  per_class_τ: True
  ortho_reg   : 0.0
  Heads       : ['avm', 'knn', 'fisher']  fusion=static

  AVM branches (global view):
    tanh_arccos : 0.3563
    boundary    : 0.3563
    circular    : 0.2874
  KNN branches (frozen local view):
    linear      : 0.3333
    shape       : 0.3333
    quadratic   : 0.3333
  Fusion weights:
    avm         : 0.3333
    knn         : 0.3333
    fisher      : 0.3333

  Feature weights (top 5):
    petal width (cm)             1.0049
    petal length (cm)            1.0049
    sepal width (cm)             0.9951
    sepal length (cm)            0.9951
  Feature weights (bottom 2):
    sepal length (cm)            0.9951
    sepal width (cm)             0.9951

  Centroid drift:
    class 0: 0.0098  
    class 1: 0.0098  
    class 2: 0.0095  

============================================================
Wine  (178 samples, 13 features, 3 classes)
============================================================
  Accuracy    : 0.9830 ± 0.0228
  Macro F1    : 0.9831 ± 0.0226
  Weighted F1 : 0.9829 ± 0.0231

Model summary (last fold):
  Classes (3): 0: 33%  1: 39%  2: 28%
  Train / val  : 123 / 20

  Lambda  : 0.600
  Tau     : cls0:0.655  cls1:0.572  cls2:0.562
  Val F1  : 0.9489

  Mahalanobis : full RDA  (alpha=0.5  ridge=1e-06)
  Ordinal EMD : False  (weight=0.5)
  SupCon      : True  (weight=0.3  temp=0.1)
  EMA         : False  (beta=0.9)
  Dual bnd    : False
  Boundary init: True
  n_prototypes: 1  per_class_τ: True
  ortho_reg   : 0.0
  Heads       : ['avm', 'knn', 'fisher']  fusion=static

  AVM branches (global view):
    tanh_arccos : 0.3503
    boundary    : 0.3564
    circular    : 0.2933
  KNN branches (frozen local view):
    linear      : 0.3333
    shape       : 0.3333
    quadratic   : 0.3333
  Fusion weights:
    avm         : 0.3333
    knn         : 0.3333
    fisher      : 0.3333

  Feature weights (top 5):
    flavanoids                   1.1850
    color_intensity              1.1686
    proline                      1.0446
    od280/od315_of_diluted_wines 1.0358
    alcalinity_of_ash            1.0297
  Feature weights (bottom 2):
    nonflavanoid_phenols         0.9055
    total_phenols                0.9155

  Centroid drift:
    class 0: 0.2786  ███████████
    class 1: 0.4639  ██████████████████
    class 2: 0.3874  ███████████████

  Lambda search: floor=0.5  chosen=0.600

============================================================
Breast Cancer  (569 samples, 30 features, 2 classes)
============================================================
  Accuracy    : 0.9631 ± 0.0140
  Macro F1    : 0.9597 ± 0.0157
  Weighted F1 : 0.9627 ± 0.0143

Model summary (last fold):
  Classes (2): 0: 37%  1: 63%
  Train / val  : 389 / 67

  Lambda  : 0.750
  Tau     : cls0:0.624  cls1:0.589
  Val F1  : 0.8454

  Mahalanobis : full RDA  (alpha=0.5  ridge=1e-06)
  Ordinal EMD : True  (weight=0.5)
  SupCon      : True  (weight=0.3  temp=0.1)
  EMA         : False  (beta=0.9)
  Dual bnd    : False
  Boundary init: True
  n_prototypes: 1  per_class_τ: True
  ortho_reg   : 0.0
  Heads       : ['avm', 'knn', 'fisher']  fusion=confidence

  AVM branches (global view):
    tanh_arccos : 0.3591
    boundary    : 0.3513
    circular    : 0.2896
  KNN branches (frozen local view):
    linear      : 0.2000
    shape       : 0.2000
    quadratic   : 0.2000
    log         : 0.2000
    rank        : 0.2000

  Feature weights (top 5):
    area error                   1.0306
    worst area                   1.0294
    worst concavity              1.0290
    mean concavity               1.0286
    mean area                    1.0285
  Feature weights (bottom 2):
    worst concave points         0.9738
    mean symmetry                0.9740

  Centroid drift:
    class 0: 0.1620  ██████
    class 1: 0.1614  ██████

  Lambda search: floor=0.5  chosen=0.750

============================================================
Red Wine Quality  (1599 samples, 11 features, 3 classes)
============================================================
  Accuracy    : 0.8337 ± 0.0311
  Macro F1    : 0.5904 ± 0.0460
  Weighted F1 : 0.8343 ± 0.0233

Model summary (last fold):
  Classes (3): 0: 4%  1: 82%  2: 14%
  Train / val  : 1090 / 190

  Lambda  : 0.800
  Tau     : cls0:0.480  cls1:0.763  cls2:0.728
  Val F1  : 0.5471

  Mahalanobis : full RDA  (alpha=0.5  ridge=1e-06)
  Ordinal EMD : True  (weight=0.5)
  SupCon      : True  (weight=0.3  temp=0.1)
  EMA         : True  (beta=0.9)
  Dual bnd    : False
  Boundary init: True
  n_prototypes: 2  per_class_τ: True
  ortho_reg   : 0.0
  Heads       : ['avm', 'knn']  fusion=legacy

  AVM branches (global view):
    tanh_arccos : 0.3732
    boundary    : 0.3178
    circular    : 0.3090
  KNN branches (frozen local view):
    linear      : 0.2000
    shape       : 0.2000
    quadratic   : 0.2000
    log         : 0.2000
    rank        : 0.2000

  Feature weights (top 5):
    volatile acidity             1.0609
    sulphates                    1.0572
    alcohol                      1.0341
    total sulfur dioxide         1.0122
    chlorides                    1.0052
  Feature weights (bottom 2):
    free sulfur dioxide          0.9555
    citric acid                  0.9594

  Centroid drift:
    class 0: 0.2670  ██████████
    class 1: 0.6150  ████████████████████████
    class 2: 0.8885  ███████████████████████████████████
  Per-prototype drift:
    class 0: p0=0.1319  p1=0.4021
    class 1: p0=0.1250  p1=1.1051
    class 2: p0=0.1283  p1=1.6488

  Lambda search: floor=0.5  chosen=0.800

============================================================
White Wine Quality  (4898 samples, 11 features, 3 classes)
============================================================
  Accuracy    : 0.6952 ± 0.0118
  Macro F1    : 0.6916 ± 0.0124
  Weighted F1 : 0.6951 ± 0.0120

Model summary (last fold):
  Classes (3): 0: 33%  1: 45%  2: 22%
  Train / val  : 3333 / 586

  Lambda  : 0.900
  Tau     : cls0:0.645  cls1:0.563  cls2:0.632
  Val F1  : 0.3476

  Mahalanobis : full RDA  (alpha=0.5  ridge=1e-06)
  Ordinal EMD : True  (weight=0.5)
  SupCon      : True  (weight=0.3  temp=0.1)
  EMA         : True  (beta=0.9)
  Dual bnd    : False
  Boundary init: True
  n_prototypes: 1  per_class_τ: True
  ortho_reg   : 0.0
  Heads       : ['avm', 'knn', 'fisher']  fusion=confidence

  AVM branches (global view):
    tanh_arccos : 0.3608
    boundary    : 0.3434
    circular    : 0.2958
  KNN branches (frozen local view):
    linear      : 0.2000
    shape       : 0.2000
    quadratic   : 0.2000
    log         : 0.2000
    rank        : 0.2000

  Feature weights (top 5):
    density                      1.0462
    alcohol                      1.0377
    chlorides                    1.0370
    volatile acidity             1.0252
    free sulfur dioxide          0.9967
  Feature weights (bottom 2):
    residual sugar               0.9507
    sulphates                    0.9695

  Centroid drift:
    class 0: 0.1793  ███████
    class 1: 0.0885  ███
    class 2: 0.1579  ██████

  Lambda search: floor=0.5  chosen=0.900
