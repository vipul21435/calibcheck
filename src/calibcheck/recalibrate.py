"""Recalibration maps, always fitted on one split and scored on another.

Four maps from a confidence to a recalibrated confidence, as described by Guo et al.
(2017), "On Calibration of Modern Neural Networks", arXiv:1706.04599:

* temperature scaling: ``sigmoid(logit(c) / T)`` for a single confidence, or
  ``softmax(z / T)`` for a vector of logits, with ``T`` minimising the log loss over
  [0.01, 100] (a fit that ends on an edge of that range is flagged with a warning);
* Platt scaling: ``sigmoid(a * logit(c) + b)`` with ``(a, b)`` by maximum likelihood;
* isotonic regression: the non-decreasing step function closest to the outcomes in
  squared error (pool adjacent violators), interpolated linearly between its points;
* histogram binning: the accuracy of the bin the confidence falls in.

A map scored on the items it was fitted on always looks good, so ``cross_fit`` splits the
items into folds, fits on all folds but one and predicts the held-out fold.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from .binning import Strategy, assign_bins
from .metrics import DEFAULT_BINS, as_arrays, bin_stats
from .multiclass import softmax

EPS = 1e-12
T_MIN, T_MAX = 1e-2, 1e2


class Calibrator(Protocol):
    """A map from confidence to recalibrated confidence, fitted on outcomes."""

    def fit(self, confidence: ArrayLike, correct: ArrayLike) -> Calibrator: ...

    def predict(self, confidence: ArrayLike) -> np.ndarray: ...


def logit(p: ArrayLike) -> np.ndarray:
    """``log(p / (1 - p))`` with ``p`` clipped to [1e-12, 1 - 1e-12]."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: ArrayLike) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _binary_nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def _golden_section(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-9) -> float:
    """Minimise a unimodal function on [lo, hi]."""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def at_temperature_bound(temperature: float) -> bool:
    """Whether a fitted temperature sits on the edge of the search range [0.01, 100]."""
    return temperature <= T_MIN * (1.0 + 1e-6) or temperature >= T_MAX * (1.0 - 1e-6)


def _fit_log_temperature(loss: Callable[[float], float]) -> float:
    """Minimise ``loss(T)`` over ``T`` in [``T_MIN``, ``T_MAX``] = [0.01, 100].

    The log loss is convex in ``1 / T``, hence unimodal in ``log T``. When it keeps falling
    towards an end of the range (all answers right or all wrong pushes ``T`` down, a
    confidence that says nothing about correctness pushes it up) the search stops at that
    end. Such a value is a limit of the search, not a fitted temperature, so a
    ``RuntimeWarning`` says so.
    """
    t = math.exp(_golden_section(lambda u: loss(math.exp(u)), math.log(T_MIN), math.log(T_MAX)))
    best = loss(t)
    for edge in (T_MIN, T_MAX):
        # Also catches a loss that is flat near an edge because probabilities are clipped.
        if loss(edge) <= best + 1e-12:
            warnings.warn(
                f"temperature search stopped at the edge of its range [{T_MIN:g}, {T_MAX:g}] "
                f"(T = {edge:g}): the log loss is as low there as anywhere inside, so this "
                "is not a fitted temperature",
                RuntimeWarning,
                stacklevel=3,
            )
            return edge
    return t


class TemperatureScaling:
    """``c -> sigmoid(logit(c) / T)``; ``T > 1`` pulls confidences towards one half.

    ``T`` is searched in [0.01, 100]; a search that ends on either edge raises a
    ``RuntimeWarning`` (see ``at_temperature_bound``).
    """

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, confidence: ArrayLike, correct: ArrayLike) -> TemperatureScaling:
        c, y = as_arrays(confidence, correct)
        z = logit(c)
        self.temperature = _fit_log_temperature(lambda t: _binary_nll(sigmoid(z / t), y))
        return self

    def predict(self, confidence: ArrayLike) -> np.ndarray:
        return sigmoid(logit(confidence) / self.temperature)


class PlattScaling:
    """``c -> sigmoid(a * logit(c) + b)``, fitted by Newton's method on the log loss."""

    def __init__(self) -> None:
        self.a, self.b = 1.0, 0.0

    def fit(self, confidence: ArrayLike, correct: ArrayLike) -> PlattScaling:
        c, y = as_arrays(confidence, correct)
        x = logit(c)
        a, b = 1.0, 0.0
        loss = _binary_nll(sigmoid(a * x + b), y)
        for _ in range(100):
            p = sigmoid(a * x + b)
            w = p * (1.0 - p)
            g_a, g_b = float(np.sum((p - y) * x)), float(np.sum(p - y))
            h_aa, h_ab, h_bb = float(np.sum(w * x * x)), float(np.sum(w * x)), float(np.sum(w))
            det = h_aa * h_bb - h_ab * h_ab
            if det <= 1e-12 * max(1.0, h_aa * h_bb):
                break
            step_a = (h_bb * g_a - h_ab * g_b) / det
            step_b = (h_aa * g_b - h_ab * g_a) / det
            scale = 1.0
            while scale > 1e-6:
                new_a, new_b = a - scale * step_a, b - scale * step_b
                new_loss = _binary_nll(sigmoid(new_a * x + new_b), y)
                if new_loss <= loss:
                    break
                scale /= 2.0
            else:
                break
            moved = max(abs(new_a - a), abs(new_b - b))
            a, b, loss = new_a, new_b, new_loss
            if moved < 1e-10:
                break
        self.a, self.b = a, b
        return self

    def predict(self, confidence: ArrayLike) -> np.ndarray:
        return sigmoid(self.a * logit(confidence) + self.b)


def pool_adjacent_violators(y: ArrayLike, weight: ArrayLike | None = None) -> np.ndarray:
    """Weighted least-squares fit of a non-decreasing sequence to ``y``."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if weight is None else np.asarray(weight, dtype=float)
    means: list[float] = []
    weights: list[float] = []
    sizes: list[int] = []
    for value, wt in zip(y.tolist(), w.tolist(), strict=True):
        means.append(value)
        weights.append(wt)
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            total = weights[-2] + weights[-1]
            means[-2] = (means[-2] * weights[-2] + means[-1] * weights[-1]) / total
            weights[-2] = total
            sizes[-2] += sizes[-1]
            del means[-1], weights[-1], sizes[-1]
    return np.repeat(np.array(means), sizes)


class IsotonicRegression:
    """Non-decreasing map fitted by pool adjacent violators.

    Equal confidences are pooled first. Between fitted points the map is linear; outside
    the fitted range it is constant.
    """

    def __init__(self) -> None:
        self.x_ = np.array([0.0, 1.0])
        self.y_ = np.array([0.0, 1.0])

    def fit(self, confidence: ArrayLike, correct: ArrayLike) -> IsotonicRegression:
        c, y = as_arrays(confidence, correct)
        x, inverse, counts = np.unique(c, return_inverse=True, return_counts=True)
        mean_y = np.bincount(inverse, weights=y) / counts
        self.x_ = x
        self.y_ = np.clip(pool_adjacent_violators(mean_y, counts.astype(float)), 0.0, 1.0)
        return self

    def predict(self, confidence: ArrayLike) -> np.ndarray:
        return np.interp(np.asarray(confidence, dtype=float), self.x_, self.y_)


class HistogramBinning:
    """Replace a confidence by the accuracy of its bin on the fitting split.

    Bins are equal-mass by default, so none is empty on the fitting split; an empty
    equal-width bin maps to its midpoint.
    """

    def __init__(self, n_bins: int = DEFAULT_BINS, strategy: Strategy = "mass") -> None:
        self.n_bins = n_bins
        self.strategy = strategy
        self.edges_ = np.array([0.0, 1.0])
        self.value_ = np.array([0.5])

    def fit(self, confidence: ArrayLike, correct: ArrayLike) -> HistogramBinning:
        stats = bin_stats(confidence, correct, n_bins=self.n_bins, strategy=self.strategy)
        value = stats.accuracy.copy()
        empty = ~stats.nonempty
        value[empty] = ((stats.edges[:-1] + stats.edges[1:]) / 2.0)[empty]
        self.edges_, self.value_ = stats.edges, value
        return self

    def predict(self, confidence: ArrayLike) -> np.ndarray:
        return self.value_[assign_bins(confidence, self.edges_)]


METHODS: dict[str, Callable[[], Calibrator]] = {
    "temperature": TemperatureScaling,
    "platt": PlattScaling,
    "isotonic": IsotonicRegression,
    "histogram": HistogramBinning,
}


def make_calibrator(method: str) -> Calibrator:
    """A fresh calibrator by name: temperature, platt, isotonic or histogram."""
    try:
        return METHODS[method]()
    except KeyError:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(METHODS)}") from None


def fold_ids(
    n: int, n_folds: int = 5, *, seed: int = 0, groups: Sequence[object] | None = None
) -> np.ndarray:
    """Assign each item to one of ``n_folds`` folds at random, keeping groups together."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    rng = np.random.default_rng(seed)
    if groups is None:
        units = np.arange(n)
        n_units = n
    else:
        if len(groups) != n:
            raise ValueError("groups must have one entry per item")
        _, units = np.unique(np.asarray([str(g) for g in groups]), return_inverse=True)
        n_units = int(units.max()) + 1
    if n_units < n_folds:
        raise ValueError(f"cannot make {n_folds} folds from {n_units} items or groups")
    fold_of_unit = np.empty(n_units, dtype=int)
    fold_of_unit[rng.permutation(n_units)] = np.arange(n_units) % n_folds
    return fold_of_unit[units]


def cross_fit(
    method: str,
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_folds: int = 5,
    seed: int = 0,
    groups: Sequence[object] | None = None,
) -> np.ndarray:
    """Out-of-fold recalibrated confidences: every item is mapped by a model fitted on the
    other folds, never on itself."""
    c, y = as_arrays(confidence, correct)
    folds = fold_ids(c.size, n_folds, seed=seed, groups=groups)
    out = np.empty_like(c)
    for f in range(n_folds):
        held_out = folds == f
        model = make_calibrator(method).fit(c[~held_out], y[~held_out])
        out[held_out] = model.predict(c[held_out])
    return np.clip(out, 0.0, 1.0)


def fit_apply(
    method: str,
    fit_confidence: ArrayLike,
    fit_correct: ArrayLike,
    confidence: ArrayLike,
) -> np.ndarray:
    """Fit on one set of items and map the confidences of another."""
    model = make_calibrator(method).fit(fit_confidence, fit_correct)
    return np.clip(model.predict(np.asarray(confidence, dtype=float)), 0.0, 1.0)


def _vector_nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    p = softmax(logits / temperature)
    return float(-np.mean(np.log(np.clip(p[np.arange(labels.size), labels], EPS, 1.0))))


def fit_temperature(logits: ArrayLike, labels: ArrayLike) -> float:
    """Temperature ``T`` minimising the log loss of ``softmax(logits / T)``.

    For probability vectors pass their logarithm as logits. Padding with ``-inf`` is
    allowed for vectors of different lengths.
    """
    z = np.atleast_2d(np.asarray(logits, dtype=float))
    y = np.asarray(labels, dtype=int)
    if y.shape != (z.shape[0],):
        raise ValueError("labels must hold one class index per item")
    return _fit_log_temperature(lambda t: _vector_nll(z, y, t))


def apply_temperature(logits: ArrayLike, temperature: float) -> np.ndarray:
    """``softmax(logits / temperature)`` row by row."""
    return softmax(np.atleast_2d(np.asarray(logits, dtype=float)) / temperature)


def cross_fit_temperature(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    n_folds: int = 5,
    seed: int = 0,
    groups: Sequence[object] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold temperature-scaled probabilities and the temperature of each fold."""
    z = np.atleast_2d(np.asarray(logits, dtype=float))
    y = np.asarray(labels, dtype=int)
    folds = fold_ids(z.shape[0], n_folds, seed=seed, groups=groups)
    out = np.empty_like(z)
    temperatures = []
    for f in range(n_folds):
        held_out = folds == f
        t = fit_temperature(z[~held_out], y[~held_out])
        temperatures.append(t)
        out[held_out] = apply_temperature(z[held_out], t)
    return out, temperatures
