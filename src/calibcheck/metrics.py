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
  arXiv:1909.10155, Definitions 5.1 and 5.2). The same paper debiases the ECE with a
  normal approximation of each bin accuracy (its Definition G.3); see ``BinStats.ece``.
* The Brier score (Brier, 1950) is ``mean((c_i - y_i) ** 2)``. Murphy (1973) splits it
  into reliability - resolution + uncertainty for forecasts that are constant within a
  bin; the reliability term is the plug-in squared calibration error.
* AUROC, the risk-coverage curve and its area (AURC) measure how well the confidence
  ranks right answers above wrong ones (selective prediction, Geifman and El-Yaniv, 2017,
  arXiv:1705.08500). They do not depend on calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .binning import Strategy, assign_bins, make_edges

DEFAULT_BINS = 15
DEFAULT_COVERAGE = 0.8


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


def folded_normal_mean(mu: ArrayLike, sd: ArrayLike) -> np.ndarray:
    """``E|X|`` for ``X ~ N(mu, sd ** 2)``, elementwise; equals ``|mu|`` where ``sd == 0``."""
    mu = np.abs(np.asarray(mu, dtype=float))
    sd = np.asarray(sd, dtype=float)
    out = mu.copy()
    pos = sd > 0
    z = mu[pos] / sd[pos]
    erf = np.array([math.erf(v / math.sqrt(2.0)) for v in z])
    out[pos] = sd[pos] * math.sqrt(2.0 / math.pi) * np.exp(-0.5 * z * z) + mu[pos] * erf
    return out


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

    def ece(self, debias: bool = False) -> float:
        """Expected calibration error over the non-empty bins, plug-in or debiased.

        The debiased value follows Definition G.3 of Kumar, Liang and Ma (2019): redraw
        each bin accuracy from ``N(acc_b, acc_b * (1 - acc_b) / n_b)``, take the expected
        plug-in ECE of the redrawn bins, and subtract the excess over the observed
        plug-in value. The expectation is the mean of a folded normal, so it is computed
        in closed form here instead of by simulation. The paper calls this a heuristic;
        it reduces the bias and does not remove it. A bin whose accuracy is 0 or 1 has
        zero estimated variance and gets no correction. The result can be negative.
        """
        m = self.nonempty
        w = self.weight[m]
        gap = np.abs(self.gap[m])
        plugin = float(np.sum(w * gap))
        if not debias:
            return plugin
        acc = self.accuracy[m]
        sd = np.sqrt(acc * (1.0 - acc) / self.count[m])
        expected = float(np.sum(w * folded_normal_mean(gap, sd)))
        return plugin - (expected - plugin)

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
    debias: bool = False,
) -> float:
    """Expected calibration error, ``sum_b (n_b / n) * |acc_b - conf_b|``.

    ``debias=True`` subtracts an estimate of the bias (see ``BinStats.ece``).
    """
    return bin_stats(confidence, correct, n_bins=n_bins, strategy=strategy).ece(debias=debias)


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


def log_loss(confidence: ArrayLike, correct: ArrayLike, *, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the outcomes, confidences clipped to [eps, 1 - eps].

    Without the clip one wrong answer given with confidence 1 makes the loss infinite.
    """
    c, y = as_arrays(confidence, correct)
    p = np.clip(c, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def auroc(confidence: ArrayLike, correct: ArrayLike) -> float:
    """Probability that a right answer has higher confidence than a wrong one.

    Ties count one half. NaN when all answers are right or all are wrong.
    """
    c, y = as_arrays(confidence, correct)
    n_pos = float(y.sum())
    n_neg = float(y.size) - n_pos
    if n_pos == 0.0 or n_neg == 0.0:
        return float("nan")
    _, inverse, counts = np.unique(c, return_inverse=True, return_counts=True)
    rank = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]
    return float((rank[y == 1.0].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


@dataclass(frozen=True)
class RiskCoverage:
    """Selective risk at every coverage ``k / n``, most confident items first.

    ``risk[k - 1]`` is the error rate among the ``k`` most confident items. Items with
    equal confidence cannot be ordered, so within a run of ties the errors are spread
    evenly: the value is the expectation over a random order of the tied items, which
    makes the curve independent of the input order.
    """

    coverage: np.ndarray
    risk: np.ndarray

    @property
    def aurc(self) -> float:
        """Area under the risk-coverage curve: the mean risk over all coverages."""
        return float(np.mean(self.risk))

    def accuracy_at(self, coverage: float) -> float:
        """Accuracy on the most confident ``ceil(coverage * n)`` items."""
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage must lie in (0, 1]")
        k = max(1, math.ceil(coverage * self.risk.size - 1e-9))
        return float(1.0 - self.risk[k - 1])


def risk_coverage(confidence: ArrayLike, correct: ArrayLike) -> RiskCoverage:
    """Risk-coverage curve of answering only the most confident items."""
    c, y = as_arrays(confidence, correct)
    n = c.size
    order = np.argsort(-c, kind="stable")
    c_sorted = c[order]
    errors = 1.0 - y[order]
    new_run = np.concatenate(([True], c_sorted[1:] != c_sorted[:-1]))
    run = np.cumsum(new_run) - 1
    run_size = np.bincount(run)
    run_errors = np.bincount(run, weights=errors)
    run_start = np.concatenate(([0], np.cumsum(run_size)[:-1]))
    errors_before = np.concatenate(([0.0], np.cumsum(run_errors)[:-1]))
    position = np.arange(n) - run_start[run] + 1
    expected_errors = errors_before[run] + position * run_errors[run] / run_size[run]
    k = np.arange(1, n + 1)
    return RiskCoverage(coverage=k / n, risk=expected_errors / k)


def aurc(confidence: ArrayLike, correct: ArrayLike) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    return risk_coverage(confidence, correct).aurc


def accuracy_at_coverage(
    confidence: ArrayLike, correct: ArrayLike, coverage: float = DEFAULT_COVERAGE
) -> float:
    """Accuracy when only the most confident share ``coverage`` of the items is answered."""
    return risk_coverage(confidence, correct).accuracy_at(coverage)


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
    coverage: float = DEFAULT_COVERAGE,
) -> dict[str, float]:
    """Every metric for a confidence and a correctness column, as a flat dictionary."""
    c, y = as_arrays(confidence, correct)
    stats = bin_stats(c, y, n_bins=n_bins, strategy=strategy)
    dec = _decompose(c, y, stats)
    sq_plugin = stats.squared_ce(debias=False)
    sq_debiased = stats.squared_ce(debias=True)
    curve = risk_coverage(c, y)
    return {
        "n": float(c.size),
        "accuracy": float(np.mean(y)),
        "mean_confidence": float(np.mean(c)),
        "ece": stats.ece(),
        "ece_debiased": stats.ece(debias=True),
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
        "log_loss": log_loss(c, y),
        "auroc": auroc(c, y),
        "aurc": curve.aurc,
        "accuracy_at_coverage": curve.accuracy_at(coverage),
    }
