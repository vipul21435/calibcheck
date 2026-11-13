"""Calibration metrics for one confidence and one 0/1 correctness per item.

Notation: item ``i`` has confidence ``c_i`` in [0, 1] and outcome ``y_i`` in {0, 1}. A
binning splits the ``n`` items into bins; bin ``b`` has ``n_b`` items, mean confidence
``conf_b`` and accuracy ``acc_b`` (the mean of ``y``).

* ECE (plug-in) is ``sum_b (n_b / n) * |acc_b - conf_b|`` and MCE is ``max_b |acc_b -
  conf_b|``, as defined by Guo et al. (2017), "On Calibration of Modern Neural Networks",
  arXiv:1706.04599.
* The squared calibration error is ``sum_b (n_b / n) * (conf_b - acc_b) ** 2``. Its
  plug-in estimate is biased upward: even a calibrated model shows ``acc_b != conf_b`` in
  a finite sample. The debiased estimate subtracts ``acc_b * (1 - acc_b) / (n_b - 1)``
  from each squared gap (Kumar, Liang and Ma, 2019, "Verified Uncertainty Calibration",
  arXiv:1909.10155, Definitions 5.1 and 5.2).
* The Brier score (Brier, 1950) is ``mean((c_i - y_i) ** 2)``. Murphy (1973) splits it
  into reliability - resolution + uncertainty for forecasts that are constant within a
  bin; the reliability term is the plug-in squared calibration error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .binning import Strategy, assign_bins, make_edges

DEFAULT_BINS = 15


def as_arrays(confidence: ArrayLike, correct: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Validate the two columns and return them as float arrays of equal length."""
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct)
    if c.ndim != 1:
        raise ValueError("confidence must be one-dimensional")
    if y.shape != c.shape:
        raise ValueError(
            f"confidence and correct must have the same length, got {c.shape} and {y.shape}"
        )
    if c.size == 0:
        raise ValueError("no items")
    if not np.all(np.isfinite(c)):
        raise ValueError("confidence contains NaN or infinite values")
    if np.any(c < 0.0) or np.any(c > 1.0):
        raise ValueError("confidence must lie in [0, 1]")
    try:
        y = y.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("correct must contain only 0/1 or booleans") from exc
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("correct must contain only 0/1 or booleans")
    return c, y


@dataclass(frozen=True)
class BinStats:
    """Counts and means per bin: all that the binned metrics need."""

    edges: np.ndarray
    count: np.ndarray
    confidence: np.ndarray
    accuracy: np.ndarray
    strategy: str

    @property
    def n_bins(self) -> int:
        return int(self.count.size)

    @property
    def n_items(self) -> int:
        return int(self.count.sum())

    @property
    def nonempty(self) -> np.ndarray:
        return self.count > 0

    @property
    def weight(self) -> np.ndarray:
        """Share of the items in each bin, ``n_b / n``."""
        return self.count / self.count.sum()

    @property
    def gap(self) -> np.ndarray:
        """``acc_b - conf_b``; negative when the bin is overconfident, NaN when empty."""
        return self.accuracy - self.confidence

    def ece(self) -> float:
        """Plug-in expected calibration error over the non-empty bins."""
        m = self.nonempty
        return float(np.sum(self.weight[m] * np.abs(self.gap[m])))

    def mce(self) -> float:
        """Largest absolute gap over the non-empty bins."""
        return float(np.max(np.abs(self.gap[self.nonempty])))

    def squared_ce(self, debias: bool = False) -> float:
        """Squared calibration error, plug-in or debiased.

        The debiased value can be negative: it is unbiased only if it is allowed to fall
        below zero when the true error is near zero. A bin with one item has no estimate
        of its sampling variance, so it contributes zero to the debiased value (its
        plug-in squared gap would be noise); ``singleton_bins`` counts such bins.
        """
        m = self.nonempty
        gap2 = self.gap[m] ** 2
        if not debias:
            return float(np.sum(self.weight[m] * gap2))
        n_b = self.count[m].astype(float)
        acc = self.accuracy[m]
        corrected = np.zeros_like(gap2)
        ok = n_b >= 2
        corrected[ok] = gap2[ok] - acc[ok] * (1.0 - acc[ok]) / (n_b[ok] - 1.0)
        return float(np.sum(self.weight[m] * corrected))

    @property
    def singleton_bins(self) -> int:
        return int(np.sum(self.count == 1))


def bin_stats(
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
) -> BinStats:
    """Bin the items and return the count, mean confidence and accuracy of each bin.

    Empty bins (possible with equal-width bins) keep a count of zero and NaN means.
    """
    c, y = as_arrays(confidence, correct)
    edges = make_edges(c, n_bins, strategy)
    idx = assign_bins(c, edges)
    size = edges.size - 1
    count = np.bincount(idx, minlength=size)
    conf = np.full(size, np.nan)
    acc = np.full(size, np.nan)
    m = count > 0
    conf[m] = np.bincount(idx, weights=c, minlength=size)[m] / count[m]
    acc[m] = np.bincount(idx, weights=y, minlength=size)[m] / count[m]
    return BinStats(edges=edges, count=count, confidence=conf, accuracy=acc, strategy=strategy)


def ece(
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
) -> float:
    """Plug-in expected calibration error, ``sum_b (n_b / n) * |acc_b - conf_b|``."""
    return bin_stats(confidence, correct, n_bins=n_bins, strategy=strategy).ece()


def mce(
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
) -> float:
    """Maximum calibration error, ``max_b |acc_b - conf_b|`` over non-empty bins."""
    return bin_stats(confidence, correct, n_bins=n_bins, strategy=strategy).mce()


def squared_calibration_error(
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
    debias: bool = True,
) -> float:
    """Squared calibration error ``sum_b (n_b / n) * (conf_b - acc_b) ** 2``.

    With ``debias=True`` (the default) each squared gap is reduced by
    ``acc_b * (1 - acc_b) / (n_b - 1)``, the debiased estimator of Kumar, Liang and Ma
    (2019). The result is not clipped and can be negative.
    """
    stats = bin_stats(confidence, correct, n_bins=n_bins, strategy=strategy)
    return stats.squared_ce(debias=debias)


def brier_score(confidence: ArrayLike, correct: ArrayLike) -> float:
    """Mean squared difference between confidence and outcome (Brier, 1950)."""
    c, y = as_arrays(confidence, correct)
    return float(np.mean((c - y) ** 2))


@dataclass(frozen=True)
class BrierDecomposition:
    """Murphy's partition of the Brier score, plus what binning leaves over.

    ``brier == reliability - resolution + uncertainty + within_bin``. For forecasts that
    are constant within each bin ``within_bin`` is zero and the three-term identity of
    Murphy (1973) is exact; otherwise it holds the within-bin variance of the confidences
    minus twice their within-bin covariance with the outcomes.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    within_bin: float


def _decompose(c: np.ndarray, y: np.ndarray, stats: BinStats) -> BrierDecomposition:
    m = stats.nonempty
    w = stats.weight[m]
    base_rate = float(np.mean(y))
    reliability = float(np.sum(w * (stats.confidence[m] - stats.accuracy[m]) ** 2))
    resolution = float(np.sum(w * (stats.accuracy[m] - base_rate) ** 2))
    uncertainty = base_rate * (1.0 - base_rate)
    brier = float(np.mean((c - y) ** 2))
    return BrierDecomposition(
        brier=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        within_bin=brier - (reliability - resolution + uncertainty),
    )


def murphy_decomposition(
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
) -> BrierDecomposition:
    """Reliability, resolution and uncertainty of the Brier score for a binning."""
    c, y = as_arrays(confidence, correct)
    return _decompose(c, y, bin_stats(c, y, n_bins=n_bins, strategy=strategy))


def summary(
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
) -> dict[str, float]:
    """All metrics implemented so far, as a flat dictionary of floats."""
    c, y = as_arrays(confidence, correct)
    stats = bin_stats(c, y, n_bins=n_bins, strategy=strategy)
    dec = _decompose(c, y, stats)
    sq_plugin = stats.squared_ce(debias=False)
    sq_debiased = stats.squared_ce(debias=True)
    return {
        "n": float(c.size),
        "accuracy": float(np.mean(y)),
        "mean_confidence": float(np.mean(c)),
        "ece": stats.ece(),
        "mce": stats.mce(),
        "sq_ce_plugin": sq_plugin,
        "sq_ce_debiased": sq_debiased,
        "l2_ce_plugin": math.sqrt(sq_plugin),
        "l2_ce_debiased": math.sqrt(max(sq_debiased, 0.0)),
        "brier": dec.brier,
        "reliability": dec.reliability,
        "resolution": dec.resolution,
        "uncertainty": dec.uncertainty,
        "within_bin": dec.within_bin,
    }
