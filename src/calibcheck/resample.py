"""Resampling: bootstrap intervals and consistency resampling.

* ``bootstrap`` draws items with replacement (whole groups when a group column is given,
  so that items of one group stay together) and returns percentile intervals.
* ``consistency_resampling`` follows Broecker and Smith (2007), "Increasing the
  Reliability of Reliability Diagrams", Weather and Forecasting 22(3): draw confidences
  with replacement, then draw each outcome as a coin flip with that confidence. The
  surrogate data are calibrated by construction, so the spread of their bin accuracies
  shows where the observed accuracies would fall for a calibrated model, and the spread of
  their ECE gives a Monte Carlo test of calibration.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .binning import Strategy, assign_bins
from .metrics import DEFAULT_BINS, as_arrays, bin_stats


def cluster_indices(groups: Sequence[object]) -> list[np.ndarray]:
    """Row indices of each distinct group, in order of first appearance."""
    position: dict[object, list[int]] = {}
    for i, g in enumerate(groups):
        position.setdefault(g, []).append(i)
    return [np.array(rows, dtype=int) for rows in position.values()]


def resample_indices(
    n: int, rng: np.random.Generator, clusters: list[np.ndarray] | None = None
) -> np.ndarray:
    """One bootstrap draw of row indices: items, or whole clusters when given."""
    if clusters is None:
        return rng.integers(0, n, size=n)
    picks = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[j] for j in picks])


def _quantiles(values: np.ndarray, level: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(finite, [(1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0])
    return float(lo), float(hi)


def bootstrap(
    statistic: Callable[[np.ndarray], dict[str, float]],
    n: int,
    *,
    n_resamples: int = 1000,
    level: float = 0.95,
    groups: Sequence[object] | None = None,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals for every entry of ``statistic(indices)``.

    ``statistic`` receives an array of row indices (with repeats) and returns a
    dictionary of numbers. With ``groups`` the draw is over whole groups (a cluster
    bootstrap). Resamples in which an entry is NaN are left out for that entry.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    if groups is not None and len(groups) != n:
        raise ValueError("groups must have one entry per item")
    clusters = cluster_indices(groups) if groups is not None else None
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_resamples):
        for key, value in statistic(resample_indices(n, rng, clusters)).items():
            draws.setdefault(key, []).append(value)
    return {key: _quantiles(np.array(values, dtype=float), level) for key, values in draws.items()}


@dataclass(frozen=True)
class Consistency:
    """Result of consistency resampling for one binning.

    ``lower`` and ``upper`` bound, per bin, the accuracy a calibrated model would show
    (NaN for bins that are empty in the data). ``p_value`` is the share of surrogate
    data sets whose plug-in ECE is at least the observed one, counting the observed one.
    """

    edges: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    ece: float
    null_mean: float
    null_quantile: float
    p_value: float
    n_resamples: int
    level: float


def consistency_resampling(
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> Consistency:
    """Consistency bands for a reliability diagram and a test of calibration.

    The surrogates are binned with the edges of the observed data. The test statistic is
    the plug-in ECE; its bias does not matter here because the observed value is compared
    with surrogate values that carry the same bias.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    c, y = as_arrays(confidence, correct)
    stats = bin_stats(c, y, n_bins=n_bins, strategy=strategy)
    n, size = c.size, stats.n_bins
    rng = np.random.default_rng(seed)
    accuracy = np.full((n_resamples, size), np.nan)
    null_ece = np.empty(n_resamples)
    for r in range(n_resamples):
        p = c[rng.integers(0, n, size=n)]
        outcome = (rng.random(n) < p).astype(float)
        idx = assign_bins(p, stats.edges)
        count = np.bincount(idx, minlength=size)
        hits = np.bincount(idx, weights=outcome, minlength=size)
        mass = np.bincount(idx, weights=p, minlength=size)
        m = count > 0
        accuracy[r, m] = hits[m] / count[m]
        null_ece[r] = np.sum(np.abs(hits[m] - mass[m])) / n
    tail = (1.0 - level) / 2.0
    lower = np.full(size, np.nan)
    upper = np.full(size, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        seen = stats.nonempty & np.any(np.isfinite(accuracy), axis=0)
        lower[seen] = np.nanquantile(accuracy[:, seen], tail, axis=0)
        upper[seen] = np.nanquantile(accuracy[:, seen], 1.0 - tail, axis=0)
    observed = stats.ece()
    exceed = int(np.sum(null_ece >= observed - 1e-12))
    return Consistency(
        edges=stats.edges,
        lower=lower,
        upper=upper,
        ece=observed,
        null_mean=float(np.mean(null_ece)),
        null_quantile=float(np.quantile(null_ece, level)),
        p_value=(1 + exceed) / (1 + n_resamples),
        n_resamples=n_resamples,
        level=level,
    )
