"""Build the metrics report and render it as Markdown or JSON."""

from __future__ import annotations

import json
from typing import Any

from numpy.typing import ArrayLike

from .binning import Strategy
from .metrics import DEFAULT_BINS, as_arrays, bin_stats, summary

METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("accuracy", "accuracy"),
    ("mean_confidence", "mean confidence"),
    ("ece", "ECE, plug-in"),
    ("mce", "MCE"),
    ("sq_ce_plugin", "squared calibration error, plug-in"),
    ("sq_ce_debiased", "squared calibration error, debiased"),
    ("l2_ce_plugin", "L2 calibration error, plug-in"),
    ("l2_ce_debiased", "L2 calibration error, debiased"),
    ("brier", "Brier score"),
    ("reliability", "reliability"),
    ("resolution", "resolution"),
    ("uncertainty", "uncertainty"),
    ("within_bin", "within-bin term"),
)

BINNING_NAMES = {"width": "equal-width", "mass": "equal-mass"}


def build_report(
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
    source: str | None = None,
) -> dict[str, Any]:
    """Metrics and the per-bin table as plain Python types, ready for ``json.dumps``."""
    c, y = as_arrays(confidence, correct)
    stats = bin_stats(c, y, n_bins=n_bins, strategy=strategy)
    metrics = summary(c, y, n_bins=n_bins, strategy=strategy)
    n_items = int(metrics.pop("n"))
    bins = []
    for i in range(stats.n_bins):
        empty = stats.count[i] == 0
        bins.append(
            {
                "lower": float(stats.edges[i]),
                "upper": float(stats.edges[i + 1]),
                "count": int(stats.count[i]),
                "confidence": None if empty else float(stats.confidence[i]),
                "accuracy": None if empty else float(stats.accuracy[i]),
                "gap": None if empty else float(stats.gap[i]),
            }
        )
    return {
        "source": source,
        "n_items": n_items,
        "binning": {
            "strategy": strategy,
            "requested_bins": int(n_bins),
            "bins": stats.n_bins,
            "empty_bins": int((stats.count == 0).sum()),
            "singleton_bins": stats.singleton_bins,
        },
        "metrics": metrics,
        "bins": bins,
    }


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, allow_nan=False) + "\n"


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    binning = report["binning"]
    title = "# Calibration report"
    if report.get("source"):
        title += f": {report['source']}"
    lines = [
        title,
        "",
        f"{report['n_items']} items, {binning['bins']} "
        f"{BINNING_NAMES[binning['strategy']]} bins"
        + (f" ({binning['empty_bins']} empty)" if binning["empty_bins"] else "")
        + ".",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for key, label in METRIC_LABELS:
        lines.append(f"| {label} | {_number(report['metrics'][key])} |")
    if binning["singleton_bins"]:
        lines += [
            "",
            f"{binning['singleton_bins']} bin(s) hold a single item and add nothing to the "
            "debiased estimate.",
        ]
    lines += ["", "| bin | items | mean confidence | accuracy | gap |", "|---|---|---|---|---|"]
    last = len(report["bins"]) - 1
    for i, row in enumerate(report["bins"]):
        close = "]" if i == last else ")"
        lines.append(
            f"| [{row['lower']:.3f}, {row['upper']:.3f}{close} | {row['count']} "
            f"| {_number(row['confidence'])} | {_number(row['accuracy'])} "
            f"| {_number(row['gap'])} |"
        )
    return "\n".join(lines) + "\n"
