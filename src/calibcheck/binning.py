"""Bins for confidences in [0, 1]: equal width, or equal mass with ties kept together.

A binning is an increasing array of ``B + 1`` edges with ``edges[0] == 0`` and
``edges[-1] == 1``. Bin ``i`` holds the values ``v`` with ``edges[i] <= v < edges[i + 1]``;
the last bin also holds ``v == 1``. This is the convention of ``numpy.histogram``. Guo et
al. (2017) write the bins as ``((m - 1) / M, m / M]`` instead; the two differ only for
values that sit exactly on an edge, which matters for stated confidences such as 0.8 or
0.9, so the choice is fixed here and applied without floating-point surprises.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

Strategy = Literal["width", "mass"]

STRATEGIES: tuple[str, ...] = ("width", "mass")


def _check_n_bins(n_bins: int) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise TypeError(f"n_bins must be an integer, got {type(n_bins).__name__}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    return int(n_bins)


def equal_width_edges(n_bins: int) -> np.ndarray:
    """Edges of ``n_bins`` bins of width ``1 / n_bins`` on [0, 1].

    Edge ``i`` is the correctly rounded value of ``i / n_bins``, the same double that the
    decimal literal has. A confidence written as 0.3 therefore compares equal to the edge
    3/10 and lands in the bin that starts there. ``numpy.linspace`` does not guarantee
    this (its third edge of ten is 0.30000000000000004).
    """
    n_bins = _check_n_bins(n_bins)
    return np.arange(n_bins + 1, dtype=float) / n_bins


def equal_mass_edges(values: ArrayLike, n_bins: int) -> np.ndarray:
    """Edges of at most ``n_bins`` bins holding nearly equal numbers of ``values``.

    The sorted values are cut at positions ``k * n // n_bins``; each interior edge is the
    value found at a cut, so it is a value that occurs in the data. Items are then assigned
    by value, which keeps equal confidences in one bin whatever the input order. When a run
    of ties spans several cuts the duplicate edges collapse, so fewer than ``n_bins`` bins
    can come back. No returned bin is empty on the data the edges were built from.
    """
    n_bins = _check_n_bins(n_bins)
    v = np.sort(np.asarray(values, dtype=float).ravel())
    if v.size == 0:
        raise ValueError("cannot build equal-mass bins from zero values")
    if not np.all(np.isfinite(v)):
        raise ValueError("values contain NaN or infinite entries")
    if v[0] < 0.0 or v[-1] > 1.0:
        raise ValueError("values must lie in [0, 1]")
    positions = (np.arange(1, n_bins) * v.size) // n_bins
    cuts = np.unique(v[positions])
    cuts = cuts[cuts > v[0]]
    return np.concatenate(([0.0], cuts, [1.0]))


def make_edges(values: ArrayLike, n_bins: int, strategy: Strategy = "width") -> np.ndarray:
    """Build bin edges for ``values`` with the named strategy (``"width"`` or ``"mass"``)."""
    if strategy == "width":
        return equal_width_edges(n_bins)
    if strategy == "mass":
        return equal_mass_edges(values, n_bins)
    raise ValueError(f"unknown binning strategy {strategy!r}; expected one of {STRATEGIES}")


def assign_bins(values: ArrayLike, edges: ArrayLike) -> np.ndarray:
    """Index of the bin of each value: the number of interior edges that are <= the value."""
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be a one-dimensional array with at least two entries")
    if np.any(np.diff(edges) < 0):
        raise ValueError("edges must be non-decreasing")
    return np.searchsorted(edges[1:-1], np.asarray(values, dtype=float), side="right")
