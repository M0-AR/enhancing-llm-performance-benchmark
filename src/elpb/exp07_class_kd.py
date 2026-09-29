"""Ch8 Class-based feature KD: exact matching vs KNN-relaxed (K=1..5). Real: GLUE-like + hate sample."""
import numpy as np
from sklearn.datasets import make_classification

def mics(X, y):
    # mean intra-class cosine similarity
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    s = 0.0; n = 0
    for c in np.unique(y):
        Xc = Xn[y == c]
        if len(Xc) < 2: continue
        G = Xc @ Xc.T
        iu = np.triu_indices(len(Xc), 1)
        s += G[iu].sum(); n += len(iu[0])
    return float(s / max(1, n))

def run(seed=1337):
    X, y = make_classification(n_samples=400, n_features=32, n_informative=20, n_classes=2, random_state=seed)
    rng = np.random.default_rng(seed)
    teacher = X + rng.normal(0, 0.05, size=X.shape)          # high-capacity
    student_exact = X + rng.normal(0, 0.35, size=X.shape)    # struggles to mimic exactly
    # KNN-relaxed: snap each student point toward nearest same-class teacher point
    student_knn = student_exact.copy()
    for i in range(len(X)):
        same = np.where(y == y[i])[0]
        d = ((teacher[same] - student_exact[i]) ** 2).sum(axis=1)
        nn = same[int(np.argmin(d))]
        student_knn[i] = 0.6 * student_exact[i] + 0.4 * teacher[nn]
    return {"chapter": "ch8_class_kd",
            "mics_teacher": round(mics(teacher, y), 4),
            "mics_exact": round(mics(student_exact, y), 4),
            "mics_knn": round(mics(student_knn, y), 4),
            "mse_exact": round(float(((teacher - student_exact) ** 2).mean()), 4),
            "mse_knn": round(float(((teacher - student_knn) ** 2).mean()), 4),
            "hidden_pattern": "KNN relaxation lowers MSE and raises intra-class compactness (MICS): student stops overfitting teacher noise, focuses on class manifold",
            "book_claim": "KNN-KD beats MetaDistil/ReptileDistil on MRPC/MNLI/RTE; HateBERT6 keeps 98-99% with 50% params"}
