"""Metrics for full probability vectors: top-label and classwise calibration.

Top-label calibration looks only at the probability of the predicted class (the usual
ECE). Classwise calibration bins the probability of every class in turn and averages the
per-class errors; with equal-width bins this is the static calibration error of Nixon et
al. (2019), "Measuring Calibration in Deep Learning", arXiv:1904.01685:
``(1 / K) * sum_k sum_b (n_bk / n) * |acc_bk - conf_bk|``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .binning import Strategy
from .metrics import DEFAULT_BINS, bin_stats


def softmax(logits: ArrayLike) -> np.ndarray:
    """Row-wise softmax. Entries of ``-inf`` (padding) get probability zero."""
    z = np.atleast_2d(np.asarray(logits, dtype=float))
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def pad_vectors(vectors: Sequence[Sequence[float]], fill: float) -> tuple[np.ndarray, bool]:
    """Stack vectors of possibly different lengths into one array, padding with ``fill``.

    Returns the array and whether padding was needed (the input was ragged).
    """
    if len(vectors) == 0:
        raise ValueError("no items")
    width = max(len(v) for v in vectors)
    if width < 2:
        raise ValueError("each probability vector needs at least two entries")
    out = np.full((len(vectors), width), fill, dtype=float)
    ragged = False
    for i, v in enumerate(vectors):
        if len(v) < 2:
            raise ValueError("each probability vector needs at least two entries")
        out[i, : len(v)] = v
        ragged = ragged or len(v) != width
    return out, ragged


def check_probabilities(probs: ArrayLike, labels: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Validate an ``(n, K)`` matrix of probabilities and ``n`` integer labels.

    Rows must sum to one within 1e-6. They are not renormalised, so that a probability
    written as 0.3 stays exactly 0.3 and cannot slip across a bin edge.
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must be a matrix with at least two columns")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    total = p.sum(axis=1)
    if np.any(np.abs(total - 1.0) > 1e-6):
        raise ValueError("each probability vector must sum to 1 (pass logits otherwise)")
    y = np.asarray(labels)
    if y.shape != (p.shape[0],):
        raise ValueError("labels must hold one class index per item")
    if not np.all(np.equal(np.mod(y, 1), 0)):
        raise ValueError("labels must be integer class indices")
    y = y.astype(int)
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("labels must be class indices in [0, K)")
    return p, y


def top_label(probs: ArrayLike, labels: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Confidence (largest probability) and correctness of the predicted class.

    Ties for the largest probability go to the lowest class index.
    """
    p, y = check_probabilities(probs, labels)
    prediction = np.argmax(p, axis=1)
    confidence = np.clip(p[np.arange(p.shape[0]), prediction], 0.0, 1.0)
    return confidence, (prediction == y).astype(float)


def classwise_ece(
    probs: ArrayLike,
    labels: ArrayLike,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
    debias: bool = False,
) -> float:
    """Mean over classes of the ECE of that class's probability against ``label == k``."""
    p, y = check_probabilities(probs, labels)
    p = np.clip(p, 0.0, 1.0)
    total = 0.0
    for k in range(p.shape[1]):
        stats = bin_stats(p[:, k], (y == k).astype(float), n_bins=n_bins, strategy=strategy)
        total += stats.ece(debias=debias)
    return total / p.shape[1]


def multiclass_brier(probs: ArrayLike, labels: ArrayLike) -> float:
    """Brier's original score: mean over items of ``sum_k (p_ik - 1[y_i = k]) ** 2``."""
    p, y = check_probabilities(probs, labels)
    onehot = np.zeros_like(p)
    onehot[np.arange(p.shape[0]), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def multiclass_log_loss(probs: ArrayLike, labels: ArrayLike, *, eps: float = 1e-12) -> float:
    """Mean negative log probability of the true class, probabilities clipped at ``eps``."""
    p, y = check_probabilities(probs, labels)
    return float(-np.mean(np.log(np.clip(p[np.arange(p.shape[0]), y], eps, 1.0))))


def vector_summary(
    probs: ArrayLike,
    labels: ArrayLike,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
    classwise: bool = True,
) -> dict[str, float]:
    """Metrics that need the whole vector; ``classwise=False`` when vectors were ragged."""
    out = {
        "nll": multiclass_log_loss(probs, labels),
        "brier_multiclass": multiclass_brier(probs, labels),
    }
    if classwise:
        out["classwise_ece"] = classwise_ece(probs, labels, n_bins=n_bins, strategy=strategy)
        out["classwise_ece_debiased"] = classwise_ece(
            probs, labels, n_bins=n_bins, strategy=strategy, debias=True
        )
    return out
