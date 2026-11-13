"""Reliability diagram with consistency bands, drawn with matplotlib (optional extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .binning import Strategy
from .metrics import DEFAULT_BINS, bin_stats
from .resample import Consistency, consistency_resampling

SERIES = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
BAND = "#c9c8c2"
SURFACE = "#fcfcfb"


def reliability_diagram(
    confidence: ArrayLike,
    correct: ArrayLike,
    path: str | Path,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
    title: str | None = None,
) -> Consistency:
    """Write a reliability diagram to ``path`` (PNG) and return the consistency result.

    Upper panel: accuracy against mean confidence per bin, the diagonal of perfect
    calibration, and for each bin the range in which its accuracy would fall for a
    calibrated model (consistency resampling, Broecker and Smith, 2007). Lower panel:
    the number of items per bin.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise RuntimeError('the diagram needs matplotlib: pip install "calibcheck[plot]"') from exc

    stats = bin_stats(confidence, correct, n_bins=n_bins, strategy=strategy)
    result = consistency_resampling(
        confidence,
        correct,
        n_bins=n_bins,
        strategy=strategy,
        n_resamples=n_resamples,
        level=level,
        seed=seed,
    )
    m = stats.nonempty
    x = stats.confidence[m]
    width = np.diff(stats.edges)[m]

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(6.0, 7.8),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
        facecolor=SURFACE,
    )
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BAND)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)

    top.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    top.vlines(x, result.lower[m], result.upper[m], color=BAND, linewidth=6.0, zorder=3)
    top.plot(x, stats.accuracy[m], color=SERIES, linewidth=2.0, zorder=4)
    top.scatter(
        x, stats.accuracy[m], s=34, color=SERIES, edgecolor=SURFACE, linewidth=1.2, zorder=5
    )
    top.set_xlim(0.0, 1.0)
    top.set_ylim(0.0, 1.0)
    top.set_ylabel("accuracy", color=INK, fontsize=10)
    handles = [
        plt.Line2D([], [], color=SERIES, linewidth=2.0, marker="o", markersize=5),
        plt.Line2D([], [], color=BAND, linewidth=6.0),
        plt.Line2D([], [], color=MUTED, linewidth=1.0, linestyle=(0, (4, 3))),
    ]
    labels = ["observed", f"{level:.0%} range if calibrated", "perfect calibration"]
    top.legend(handles, labels, loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    if title:
        top.set_title(title, color=INK, fontsize=11, loc="left")

    bottom.bar(stats.edges[:-1][m], stats.count[m], width=width * 0.94, align="edge", color=SERIES)
    bottom.set_ylabel("items", color=INK, fontsize=10)
    bottom.set_xlabel("confidence", color=INK, fontsize=10)

    fig.savefig(Path(path), format="png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return result
