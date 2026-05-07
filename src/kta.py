"""
Kernel Target Alignment (KTA) optimizers for quantum kernel methods.

This module provides five strategies for aligning a parameterized quantum kernel
to a classification task via gradient-based or analytical KTA maximization:

    FullKTA           – gradient computed on the entire training set each epoch
    RandomKTA         – stochastic mini-batch sampling each epoch
    GreedyKTA         – active-learning selection of the most uncertain samples
    CentroidBasedKTA  – alternating gradient-based optimization of kernel weights and centroids
    QuackKTA          – QUACK strategy: uses full training data instead of sub-centroids
                        with gradient-based optimization of kernel weights and main centroids

All strategies share a common abstract base (BaseKTA) that houses kernel matrix
construction, SVM evaluation, centering, and the main training loop.

Alignment with PyTorch TrainModel (train_method='ccka'):
  -------------------------------------------------------
  PyTorch uses a SINGLE Adam optimizer for the KAO step that jointly updates
  both kernel weights and sub-centroids:

      self._kernel_optimizer = optim.Adam([
          {'params': self._kernel.parameters(), 'lr': self._lr},
          {'params': self._class_centroids,     'lr': self._cclr},
      ])

  The CO step uses a *separate* per-class optimizer that updates ONLY the
  main centroid for the selected class:

      self._optimizers[_class] = optim.Adam([{'params': main_centroid, 'lr': self._mclr}])

  This module replicates that behaviour exactly:
    - _kao_weight_optimizer  (lr=learning_rate)   ← kernel weights
    - _kao_sub_optimizer     (lr=sub_centroid_lr) ← sub-centroids (jointly w/ KAO)
    - _co_main_optimizer     (lr=centroid_lr)     ← main centroids only (CO step)

  CO box constraint uses relu(c-1)+relu(-c) matching PyTorch (not per-feature bounds).
  CO step uses raw sub-centroid labels, not ±1 conversion.

Backward-compatible lowercase aliases are exported at module level.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pprint import pformat
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax as ox
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix)
from sklearn.svm import SVC
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Display utilities
# ─────────────────────────────────────────────────────────────────────────────

def _print_box(title: str, lines: list[str], width: int = 78) -> None:
    bar = "─" * (width - 2)
    print(f"┌{bar}┐")
    print(f"│ {title.center(width - 4)} │")
    print(f"├{bar}┤")
    for line in lines:
        for segment in pformat(line, width=width - 6).splitlines():
            print(f"│ {segment.ljust(width - 4)} │")
    print(f"└{bar}┘\n")


def print_training_summary(history: dict[str, Any], width: int = 78) -> None:
    """Pretty-print a training history dictionary returned by ``align()``."""
    _print_box(
        "TRAINING SUMMARY",
        [
            f"Epochs run          : {len(history['loss_history'])}",
            f"Total training time : {history['time']:.2f} s",
        ],
        width,
    )
    _print_box(
        "ACCURACY METRICS",
        [
            f"Initial train accuracy : {history['init_train_accuracy']:.4f}",
            f"Final   train accuracy : {history['train_accuracy_history'][-1]:.4f}",
            f"Initial test  accuracy : {history['init_test_accuracy']:.4f}",
            f"Final   test  accuracy : {history['test_accuracy_history'][-1]:.4f}",
        ],
        width,
    )
    _print_box(
        "CLASSIFICATION METRICS (FINAL EPOCH)",
        [
            f"F1 score  : {history['f1_score_history'][-1]:.4f}",
            f"Precision : {history['precision_score_history'][-1]:.4f}",
            f"Recall    : {history['recall_score_history'][-1]:.4f}",
        ],
        width,
    )
    _print_box(
        "ALIGNMENT & OPTIMIZATION",
        [
            f"Initial alignment    : {history['alignment_history'][0]:.6f}",
            f"Final alignment    : {history['alignment_history'][-1]:.6f}",
            f"Circuit executions : {history['circuit_executions']}",
        ],
        width,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseKTA(ABC):
    """
    Abstract base class for KTA-based quantum kernel optimizers.

    Concrete subclasses implement :meth:`_get_batch` to control *which* data
    subset drives the gradient at each training step.  Everything else —
    kernel matrix construction, centering, SVM evaluation, and the training
    loop — lives here.

    Parameters
    ----------
    kernel_model :
        Object exposing ``kernel_model.circuit.init_weights()`` and
        ``kernel_model.forward(x1, x2, weights)``.
    data : jnp.ndarray, shape (N, D)
    labels : jnp.ndarray, shape (N,)
    split_size : float
        Fraction of data used for training (default 0.8).
    matrix_type : {'regular', 'nystrom'}
        How to build the kernel matrix.
    landmark_points : int
        Number of Nyström landmarks; required when *matrix_type='nystrom'*.
    centering : bool
        Apply kernel centering (H K H) before use.
    epochs : int
    learning_rate : float
    optimizer : {'adam', 'sgd'}
    """

    _OPTIMIZERS: dict[str, Any] = {"adam": ox.adam, "sgd": ox.sgd}
    _MATRIX_TYPES: frozenset[str] = frozenset({"regular", "nystrom"})

    def __init__(
        self,
        kernel_model,
        data: jnp.ndarray,
        labels: jnp.ndarray,
        *,
        split_size: float = 0.8,
        matrix_type: str = "regular",
        landmark_points: int = 0,
        centering: bool = False,
        epochs: int = 100,
        learning_rate: float = 0.01,
        optimizer: str = "adam",
        **_ignored: Any,
    ) -> None:
        # ── Validation ────────────────────────────────────────────────────
        if matrix_type not in self._MATRIX_TYPES:
            raise ValueError(
                f"matrix_type must be one of {self._MATRIX_TYPES!r}, got {matrix_type!r}"
            )
        if not (0.0 < split_size < 1.0):
            raise ValueError(f"split_size must be in (0, 1), got {split_size}")
        if matrix_type == "nystrom" and landmark_points <= 0:
            raise ValueError(
                "landmark_points must be > 0 when matrix_type='nystrom'"
            )

        # ── Store hyperparameters ─────────────────────────────────────────
        self.kernel_model = kernel_model
        self.matrix_type = matrix_type
        self.landmark_points = landmark_points
        self.centering = centering
        self.epochs = epochs
        self.split_size = split_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer.lower()

        # ── Data split ────────────────────────────────────────────────────
        self.xtrain, self.xtest, self.ytrain, self.ytest = self._split_data(
            data, labels, seed=42
        )

        # ── Weights & optimizer ───────────────────────────────────────────
        self.weights = kernel_model.circuit.init_weights()
        self._optimizer = self._build_optimizer(self.learning_rate)
        self.opt_state = self._optimizer.init(self.weights)

        # ── JIT-compiled functions ────────────────────────────────────────
        self._loss_fn = jax.jit(self._loss_kta)
        self._grad_fn = jax.jit(jax.grad(self._loss_kta))

    # ── Optimizer factory ──────────────────────────────────────────────────

    def _build_optimizer(self, lr: float) -> ox.GradientTransformation:
        if self.optimizer_name not in self._OPTIMIZERS:
            raise ValueError(
                f"Optimizer {self.optimizer_name!r} not supported. "
                f"Choose from: {list(self._OPTIMIZERS)}"
            )
        return self._OPTIMIZERS[self.optimizer_name](lr)

    # ── Data splitting ─────────────────────────────────────────────────────

    def _split_data(
        self,
        data: jnp.ndarray,
        labels: jnp.ndarray,
        seed: int = 42,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        n = len(data)
        perm = jax.random.permutation(jax.random.PRNGKey(seed), n)
        split = int(n * self.split_size)
        tr, te = perm[:split], perm[split:]
        return data[tr], data[te], labels[tr], labels[te]

    # ── Kernel matrix helpers ──────────────────────────────────────────────

    @staticmethod
    def _pairwise(
        A: jnp.ndarray, B: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Expand (A, B) so that row ``i * M + j`` of the returned arrays
        equals ``(A[i], B[j])`` — covering all N×M ordered pairs.
        Mirrors PyTorch's  x_0 = a.repeat(M, 1)  /  x_1 = b.repeat_interleave(N, dim=0).
        """
        N, M = A.shape[0], B.shape[0]
        return jnp.repeat(A, M, axis=0), jnp.tile(B, (N, 1))

    def regular_kernel_matrix(
        self, weights, X: jnp.ndarray
    ) -> jnp.ndarray:
        """Full NxN kernel matrix using upper-triangular computation."""
        N = X.shape[0]

        # Get indices for upper triangular (including diagonal)
        iu, ju = jnp.triu_indices(N)

        # Gather pairs
        x1 = X[iu]
        x2 = X[ju]

        # Compute kernel values only for upper triangle
        k_vals = self.kernel_model.forward(x1, x2, weights)

        # Initialize full matrix
        K = jnp.zeros((N, N))

        # Fill upper triangle
        K = K.at[iu, ju].set(k_vals)

        # Mirror to lower triangle
        K = K.at[ju, iu].set(k_vals)

        return K

    def nystrom_kernel_matrix(
        self, weights, X: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Nyström approximation: K ≈ K_NM · K_MM⁻¹ · K_NM^T.

        The first ``landmark_points`` rows of X are used as landmarks.
        """
        M = self.landmark_points
        if not (0 < M <= len(X)):
            raise ValueError(
                f"landmark_points={M} is invalid for data of length {len(X)}"
            )
        N = X.shape[0]
        landmarks = X[:M]

        x1, x2 = self._pairwise(X, landmarks)
        KNM = self.kernel_model.forward(x1, x2, weights).reshape(N, M)

        x1, x2 = self._pairwise(landmarks, landmarks)
        KMM = self.kernel_model.forward(x1, x2, weights).reshape(M, M)
        KMM_inv = jnp.linalg.inv(KMM + 1e-8 * jnp.eye(M))

        return KNM @ KMM_inv @ KNM.T

    def test_kernel_matrix(
        self, weights, X_train: jnp.ndarray, X_test: jnp.ndarray
    ) -> jnp.ndarray:
        """M×N cross-kernel matrix between X_test (rows) and X_train (cols).

        Mirrors PyTorch:
            x_0 = test_data.repeat_interleave(N_train, dim=0)
            x_1 = train_data.repeat(N_test, 1)
            K   = kernel(x_0, x_1).reshape(N_test, N_train)
        """
        N, M = X_train.shape[0], X_test.shape[0]
        x1 = jnp.repeat(X_test, N, axis=0)
        x2 = jnp.tile(X_train, (M, 1))
        return self.kernel_model.forward(x1, x2, weights).reshape(M, N)

    def _kernel_matrix(self, weights, X: jnp.ndarray) -> jnp.ndarray:
        """Dispatch to the configured matrix type."""
        if self.matrix_type == "regular":
            return self.regular_kernel_matrix(weights, X)
        return self.nystrom_kernel_matrix(weights, X)

    def _apply_centering(self, K: jnp.ndarray) -> jnp.ndarray:
        """Apply kernel centering H·K·H if enabled, otherwise pass through."""
        if not self.centering:
            return K
        n = K.shape[0]
        H = jnp.eye(n) - jnp.ones((n, n)) / n
        return H @ K @ H
    
    # ─────────────────────────────────────────────────────────────────────
    # Kernel diagnostics
    # ─────────────────────────────────────────────────────────────────────

    def kernel_variance(self, K: jnp.ndarray) -> float:
        return float(jnp.var(K))


    def effective_rank(self, K: jnp.ndarray, eps: float = 1e-12) -> float:
        eigvals = jnp.linalg.eigvalsh(K)
        eigvals = jnp.clip(eigvals, eps, None)

        p = eigvals / jnp.sum(eigvals)
        entropy = -jnp.sum(p * jnp.log(p))

        return float(jnp.exp(entropy))


    def frobenius_norm(self, K: jnp.ndarray) -> float:
        return float(jnp.linalg.norm(K, ord="fro"))


    def centered_alignment(
        self,
        K: jnp.ndarray,
        y: jnp.ndarray
    ) -> float:
        n = K.shape[0]
        H = jnp.eye(n) - jnp.ones((n, n)) / n

        Kc = H @ K @ H

        T = y[:, None] * y[None, :]
        Tc = H @ T @ H

        num = jnp.sum(Kc * Tc)
        denom = (
            jnp.linalg.norm(Kc, ord="fro")
            * jnp.linalg.norm(Tc, ord="fro")
        )

        return float(num / (denom + 1e-10))

    # ── KTA (full-matrix variant, used by FullKTA / RandomKTA / GreedyKTA) ─

    def alignment(
        self, weights, X: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Kernel–Target Alignment between the (centered) kernel matrix and the
        label outer product.

        Matches PyTorch _loss_ta:
            yTKy / (sqrt(trace(K²)) * N)
        which equals  <K, T>_F / (||K||_F · ||T||_F)  for ±1 labels since
        ||T||_F = ||y||² = N.
        """
        K = self._apply_centering(self._kernel_matrix(weights, X))
        T = y[:, None] * y[None, :]          # label outer product — target kernel
        norm = jnp.linalg.norm(K, ord="fro") * jnp.linalg.norm(T, ord="fro")
        return jnp.sum(K * T) / (norm + 1e-10)

    def _loss_kta(
        self, weights, X: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        return 1.0 - self.alignment(weights, X, y)

    # ── SVM evaluation ─────────────────────────────────────────────────────

    def svm_training(
        self, X: jnp.ndarray, y: jnp.ndarray
    ) -> dict[str, Any]:
        """
        Fit an SVM with the current kernel and evaluate on train + test sets.

        Returns a dict with keys: svm, train_accuracy, test_accuracy,
        f1_score, precision_score, recall_score.
        """
        K_train = np.asarray(
            self._apply_centering(self._kernel_matrix(self.weights, X))
        )
        y_train_np = np.asarray(y)
        y_test_np  = np.asarray(self.ytest)

        svm = SVC(kernel="precomputed", C=1.0, probability=True, max_iter=10_000)
        svm.fit(K_train, y_train_np)

        K_test_raw = np.asarray(
            self.test_kernel_matrix(self.weights, self.xtrain, self.xtest)
        )
        if self.centering:
            n_train = K_train.shape[0]
            train_col_means = K_train.mean(axis=0, keepdims=True)
            train_mean      = K_train.mean()
            K_test = (
                K_test_raw
                - K_test_raw.mean(axis=1, keepdims=True)
                - train_col_means
                + train_mean
            )
        else:
            K_test = K_test_raw

        y_pred_train = svm.predict(K_train)
        y_pred_test  = svm.predict(K_test)

        train_margin = float(np.mean(np.abs(svm.decision_function(K_train))))
        test_margin  = float(np.mean(np.abs(svm.decision_function(K_test))))

        train_conf = confusion_matrix(y_train_np, y_pred_train)
        test_conf  = confusion_matrix(y_test_np, y_pred_test)

        return {
            "svm": svm,

            # Accuracy
            "train_accuracy": float(
                accuracy_score(y_train_np, y_pred_train)
            ),
            "test_accuracy": float(
                accuracy_score(y_test_np, y_pred_test)
            ),

            # Balanced accuracy
            "balanced_train_accuracy": float(
                balanced_accuracy_score(y_train_np, y_pred_train)
            ),
            "balanced_test_accuracy": float(
                balanced_accuracy_score(y_test_np, y_pred_test)
            ),

            # Classification metrics
            "f1_score": float(
                f1_score(y_test_np, y_pred_test, average="macro")
            ),
            "precision_score": float(
                precision_score(y_test_np, y_pred_test, average="macro")
            ),
            "recall_score": float(
                recall_score(y_test_np, y_pred_test, average="macro")
            ),

            # Margins
            "train_margin": train_margin,
            "test_margin": test_margin,

            # Overfitting gap
            "generalization_gap": float(
                accuracy_score(y_train_np, y_pred_train)
                - accuracy_score(y_test_np, y_pred_test)
            ),

            # Confusion matrices
            "train_confusion_matrix": train_conf,
            "test_confusion_matrix": test_conf,
        }

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def _get_batch(
        self, epoch: int
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return *(X_batch, y_batch)* for the current gradient step."""

    # ── Main training loop ─────────────────────────────────────────────────

    def align(self) -> dict[str, Any]:
        """
        Run KTA optimization and return a training history dictionary.

        History keys
        ------------
        weights, init_train_accuracy, init_test_accuracy,
        alignment_history, loss_history,
        train_accuracy_history, test_accuracy_history,
        f1_score_history, precision_score_history, recall_score_history,
        time, circuit_executions
        """
        init = self.svm_training(self.xtrain, self.ytrain)

        alignment_hist: list[float] = []
        loss_hist:      list[float] = []
        train_acc, test_acc = [], []
        balanced_train_acc, balanced_test_acc = [], []

        f1s, precs, recs = [], [], []

        train_margins, test_margins = [], []
        generalization_gaps = []

        kernel_variances = []
        effective_ranks = []
        fro_norms = []
        centered_alignments = []

        start = time.perf_counter()
        desc  = f"[{type(self).__name__}] KTA alignment"

        for epoch in tqdm(range(self.epochs), desc=desc):
            X_b, y_b = self._get_batch(epoch)

            loss_hist.append(float(self._loss_fn(self.weights, X_b, y_b)))
            alignment_hist.append(
                float(self.alignment(self.weights, self.xtrain, self.ytrain))
            )

            grads = self._grad_fn(self.weights, X_b, y_b)
            updates, self.opt_state = self._optimizer.update(grads, self.opt_state)
            self.weights = ox.apply_updates(self.weights, updates)

        result = self.svm_training(self.xtrain, self.ytrain)
        K_train_analysis = self._apply_centering(self._kernel_matrix(self.weights, self.xtrain))
        kernel_variances.append(self.kernel_variance(K_train_analysis))
        effective_ranks.append(self.effective_rank(K_train_analysis))
        fro_norms.append(self.frobenius_norm(K_train_analysis))
        centered_alignments.append(self.centered_alignment(K_train_analysis, self.ytrain))
        balanced_train_acc.append(result["balanced_train_accuracy"])
        balanced_test_acc.append(result["balanced_test_accuracy"])
        train_margins.append(result["train_margin"])
        test_margins.append(result["test_margin"])
        generalization_gaps.append(result["generalization_gap"])
        train_acc.append(result["train_accuracy"])
        test_acc.append(result["test_accuracy"])
        f1s.append(result["f1_score"])
        precs.append(result["precision_score"])
        recs.append(result["recall_score"])
            

        history: dict[str, Any] = {
            "weights":                  self.weights,
            "init_train_accuracy":      init["train_accuracy"],
            "init_test_accuracy":       init["test_accuracy"],
            "alignment_history":        alignment_hist,
            "loss_history":             loss_hist,
            "train_accuracy_history":   train_acc,
            "test_accuracy_history":    test_acc,
            "f1_score_history":         f1s,
            "precision_score_history":  precs,
            "recall_score_history":     recs,
            "time":                     time.perf_counter() - start,
            "circuit_executions":       self.kernel_model.circuit_executions,
            "balanced_train_accuracy_history": balanced_train_acc,
            "balanced_test_accuracy_history": balanced_test_acc,
            "train_margin_history": train_margins,
            "test_margin_history": test_margins,
            "generalization_gap_history": generalization_gaps,
            "kernel_variance_history": kernel_variances,
            "effective_rank_history": effective_ranks,
            "fro_norm_history": fro_norms,
            "centered_alignment_history": centered_alignments,
        }
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Concrete KTA strategies — gradient-based
# ─────────────────────────────────────────────────────────────────────────────

class FullKTA(BaseKTA):
    """
    Full-batch KTA: gradient is computed over the entire training set each epoch.
    Mirrors PyTorch  train_method='full'.
    """

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.xtrain, self.ytrain


class RandomKTA(BaseKTA):
    """
    Stochastic KTA: draw a random mini-batch each epoch.
    Mirrors PyTorch  train_method='random'.

    Parameters
    ----------
    random_samples : int
        Mini-batch size (default 4).
    """

    def __init__(self, *args, random_samples: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if random_samples <= 0:
            raise ValueError("random_samples must be > 0")
        self.random_samples = random_samples
        self._rng = jax.random.PRNGKey(0)
        self._perm: jnp.ndarray | None = None
        self._ptr: int = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._rng, subkey = jax.random.split(self._rng)
        self._perm = jax.random.permutation(subkey, len(self.xtrain))
        self._ptr = 0

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self._ptr + self.random_samples > len(self.xtrain):
            self._reshuffle()
        idx = self._perm[self._ptr : self._ptr + self.random_samples]
        self._ptr += self.random_samples
        return self.xtrain[idx], self.ytrain[idx]

# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible lowercase aliases
# ─────────────────────────────────────────────────────────────────────────────

fullKTA          = FullKTA
randomKTA        = RandomKTA

__all__ = [
    "BaseKTA",
    "FullKTA",
    "RandomKTA",
    "print_training_summary",
    # aliases
    "fullKTA",
    "randomKTA",
]