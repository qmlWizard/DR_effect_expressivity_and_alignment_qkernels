# metrics_full.py

import jax.numpy as jnp
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# =========================================================
# 1. LABEL KERNEL
# =========================================================

def label_kernel(y):
    y = jnp.asarray(y)
    return jnp.outer(y, y)


# =========================================================
# 2. KTA
# =========================================================

def kta(K, y):
    K = jnp.asarray(K)
    Y = label_kernel(y)

    num = jnp.sum(K * Y)
    denom = jnp.linalg.norm(K) * jnp.linalg.norm(Y)

    return num / (denom + 1e-12)


# =========================================================
# 3. CENTERED KTA
# =========================================================

def center_matrix(K):
    n = K.shape[0]
    H = jnp.eye(n) - jnp.ones((n, n)) / n
    return H @ K @ H


def centered_kta(K, y):
    Kc = center_matrix(K)
    Yc = center_matrix(label_kernel(y))

    num = jnp.sum(Kc * Yc)
    denom = jnp.linalg.norm(Kc) * jnp.linalg.norm(Yc)

    return num / (denom + 1e-12)


# =========================================================
# 4. KERNEL VARIANCE
# =========================================================

def kernel_variance(K):
    return jnp.var(K)


# =========================================================
# 5. EIGENVALUES
# =========================================================

def kernel_eigenvalues(K):
    eigvals = jnp.linalg.eigvalsh(K)
    return jnp.sort(eigvals)[::-1]


# =========================================================
# 6. EFFECTIVE RANK
# =========================================================

def effective_rank(K, eps=1e-12):
    eigvals = jnp.linalg.eigvalsh(K)
    eigvals = jnp.clip(eigvals, eps, None)

    p = eigvals / jnp.sum(eigvals)
    entropy = -jnp.sum(p * jnp.log(p))

    return jnp.exp(entropy)


# =========================================================
# 7. FROBENIUS NORM
# =========================================================

def frobenius_norm(K):
    return jnp.linalg.norm(K)


# =========================================================
# 8. SVM TRAIN + TEST ACCURACY
# =========================================================

def svm_train_test_accuracy(K, y, test_size=0.3, seed=42, C=1.0):
    """
    Computes BOTH train and test accuracy using precomputed kernel.
    """

    K = np.array(K)
    y = np.array(y)

    # Ensure labels are {-1, +1}
    if set(np.unique(y)) == {0, 1}:
        y = 2*y - 1

    n = K.shape[0]
    indices = np.arange(n)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )

    # Kernel splits
    K_train = K[np.ix_(train_idx, train_idx)]
    K_test  = K[np.ix_(test_idx, train_idx)]

    y_train = y[train_idx]
    y_test  = y[test_idx]

    # Train SVM
    clf = SVC(kernel="precomputed", C=C, max_iter=10_000)
    clf.fit(K_train, y_train)

    # Predictions
    y_train_pred = clf.predict(K_train)
    y_test_pred  = clf.predict(K_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test, y_test_pred)

    return train_acc, test_acc


# =========================================================
# 9. FULL METRICS PIPELINE
# =========================================================

def compute_metrics(K, y, seed=42):
    train_acc, test_acc = svm_train_test_accuracy(K, y, seed=seed)

    return {
        "kta": float(kta(K, y)),
        "centered_kta": float(centered_kta(K, y)),
        "variance": float(kernel_variance(K)),
        "effective_rank": float(effective_rank(K)),
        "fro_norm": float(frobenius_norm(K)),
        "svm_train_acc": float(train_acc),
        "svm_test_acc": float(test_acc),
    }