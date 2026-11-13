"""Build the reports of the four commands and render them as Markdown or JSON."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .binning import Strategy
from .io import Items
from .metrics import (
    DEFAULT_BINS,
    DEFAULT_COVERAGE,
    as_arrays,
    bin_stats,
    risk_coverage,
    summary,
)
from .multiclass import top_label, vector_summary
from .recalibrate import (
    METHODS,
    apply_temperature,
    cross_fit,
    cross_fit_temperature,
    fit_apply,
    fit_temperature,
    make_calibrator,
)
from .resample import bootstrap
from .stated import digit_preference

METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("accuracy", "accuracy"),
    ("mean_confidence", "mean confidence"),
    ("ece", "ECE, plug-in"),
    ("ece_debiased", "ECE, debiased"),
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
    ("log_loss", "log loss"),
    ("auroc", "AUROC"),
    ("aurc", "AURC"),
    ("accuracy_at_coverage", "accuracy at {coverage:.0%} coverage"),
    ("classwise_ece", "classwise ECE, plug-in"),
    ("classwise_ece_debiased", "classwise ECE, debiased"),
    ("nll", "log loss of the full vector"),
    ("brier_multiclass", "Brier score of the full vector"),
)

RECALIBRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ece", "ECE, plug-in"),
    ("ece_debiased", "ECE, debiased"),
    ("l2_ce_debiased", "L2 CE, debiased"),
    ("brier", "Brier"),
    ("log_loss", "log loss"),
    ("auroc", "AUROC"),
)

BINNING_NAMES = {"width": "equal-width", "mass": "equal-mass"}


def _clean(value: float) -> float | None:
    """JSON has no NaN; an undefined metric becomes null."""
    return None if value is None or math.isnan(value) else float(value)


def _label(key: str, coverage: float) -> str:
    return dict(METRIC_LABELS)[key].format(coverage=coverage)


def build_report(
    confidence: ArrayLike,
    correct: ArrayLike,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
    source: str | None = None,
    coverage: float = DEFAULT_COVERAGE,
    n_resamples: int = 0,
    level: float = 0.95,
    seed: int = 0,
    groups: Sequence[object] | None = None,
    probs: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    classwise: bool = True,
    stated: bool = False,
) -> dict[str, Any]:
    """Metrics, optional bootstrap intervals and the per-bin table as plain Python types.

    With ``n_resamples > 0`` every metric gets a percentile interval from a bootstrap over
    items, or over whole groups when ``groups`` is given.
    """
    c, y = as_arrays(confidence, correct)
    stats = bin_stats(c, y, n_bins=n_bins, strategy=strategy)

    def statistic(idx: np.ndarray) -> dict[str, float]:
        out = summary(c[idx], y[idx], n_bins=n_bins, strategy=strategy, coverage=coverage)
        del out["n"]
        if probs is not None and labels is not None:
            out.update(
                vector_summary(
                    probs[idx], labels[idx], n_bins=n_bins, strategy=strategy, classwise=classwise
                )
            )
        return out

    metrics = statistic(np.arange(c.size))
    curve = risk_coverage(c, y)
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
    report: dict[str, Any] = {
        "source": source,
        "n_items": int(c.size),
        "binning": {
            "strategy": strategy,
            "requested_bins": int(n_bins),
            "bins": stats.n_bins,
            "empty_bins": int((stats.count == 0).sum()),
            "singleton_bins": stats.singleton_bins,
        },
        "coverage": coverage,
        "top_label": probs is not None,
        "metrics": {key: _clean(value) for key, value in metrics.items()},
        "bins": bins,
        "risk_coverage": [
            {"coverage": q / 10, "accuracy": curve.accuracy_at(q / 10)} for q in range(1, 11)
        ],
    }
    if n_resamples:
        bounds = bootstrap(
            statistic, c.size, n_resamples=n_resamples, level=level, groups=groups, seed=seed
        )
        report["intervals"] = {
            "level": level,
            "n_resamples": n_resamples,
            "seed": seed,
            "n_groups": None if groups is None else len(set(groups)),
            "bounds": {key: [_clean(lo), _clean(hi)] for key, (lo, hi) in bounds.items()},
        }
    if stated:
        report["stated"] = digit_preference(c)
    return report


def build_recalibration_report(
    items: Items,
    *,
    methods: Sequence[str] = tuple(METHODS),
    n_folds: int = 5,
    seed: int = 0,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
    fit_items: Items | None = None,
    in_sample: bool = False,
    source: str | None = None,
) -> dict[str, Any]:
    """Metrics before and after each recalibration map.

    Each map is cross-fitted over ``n_folds`` folds (groups kept together), or fitted on
    ``fit_items`` and applied to ``items``. With vectors, temperature scaling is also
    fitted on the whole vector. ``in_sample=True`` adds, for contrast, rows where the map
    is fitted and scored on the same items.
    """
    c, y = items.confidence, items.correct

    def row(name: str, fit: str, conf: np.ndarray, corr: np.ndarray, **extra: Any) -> dict:
        values = summary(conf, corr, n_bins=n_bins, strategy=strategy)
        return {
            "method": name,
            "fit": fit,
            **{key: _clean(values[key]) for key, _ in RECALIBRATION_COLUMNS},
            **extra,
        }

    held_out = "fitted on the other file" if fit_items is not None else f"{n_folds} folds"
    rows = [row("none", "-", c, y)]
    for method in methods:
        if fit_items is not None:
            after = fit_apply(method, fit_items.confidence, fit_items.correct, c)
        else:
            after = cross_fit(method, c, y, n_folds=n_folds, seed=seed, groups=items.group)
        rows.append(row(method, held_out, after, y))
        if in_sample:
            same = np.clip(make_calibrator(method).fit(c, y).predict(c), 0.0, 1.0)
            rows.append(row(method, "same items (optimistic)", same, y))
    if items.logits is not None and items.label is not None and "temperature" in methods:
        if fit_items is not None and fit_items.logits is not None:
            temperatures = [fit_temperature(fit_items.logits, fit_items.label)]
            scaled = apply_temperature(items.logits, temperatures[0])
        else:
            scaled, temperatures = cross_fit_temperature(
                items.logits, items.label, n_folds=n_folds, seed=seed, groups=items.group
            )
        conf, corr = top_label(scaled, items.label)
        rows.append(row("temperature on vectors", held_out, conf, corr, temperatures=temperatures))
    return {
        "source": source,
        "n_items": len(items),
        "folds": None if fit_items is not None else n_folds,
        "seed": seed,
        "grouped": items.group is not None,
        "binning": {"strategy": strategy, "requested_bins": int(n_bins)},
        "rows": rows,
    }


def build_comparison_report(
    a: Items,
    b: Items,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: Strategy = "width",
    coverage: float = DEFAULT_COVERAGE,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
    sources: tuple[str | None, str | None] = (None, None),
) -> dict[str, Any]:
    """Metrics of two models on the same items and a paired bootstrap of the differences.

    Items are matched by id when both sides have ids, by position otherwise. The same
    resampled items (or groups, taken from ``a``) are used for both models.
    """
    order = np.arange(len(b))
    if a.ids is not None and b.ids is not None:
        if set(a.ids) != set(b.ids):
            raise ValueError("the two files do not hold the same ids")
        where = {key: i for i, key in enumerate(b.ids)}
        order = np.array([where[key] for key in a.ids])
    elif len(a) != len(b):
        raise ValueError(f"the two files differ in length: {len(a)} and {len(b)} items")
    ca, ya = a.confidence, a.correct
    cb, yb = b.confidence[order], b.correct[order]

    def both(idx: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        sa = summary(ca[idx], ya[idx], n_bins=n_bins, strategy=strategy, coverage=coverage)
        sb = summary(cb[idx], yb[idx], n_bins=n_bins, strategy=strategy, coverage=coverage)
        return sa, sb

    def difference(idx: np.ndarray) -> dict[str, float]:
        sa, sb = both(idx)
        return {key: sa[key] - sb[key] for key in sa if key != "n"}

    sa, sb = both(np.arange(len(a)))
    bounds = bootstrap(
        difference, len(a), n_resamples=n_resamples, level=level, groups=a.group, seed=seed
    )
    rows = [
        {
            "metric": key,
            "a": _clean(sa[key]),
            "b": _clean(sb[key]),
            "difference": _clean(sa[key] - sb[key]),
            "interval": [_clean(bounds[key][0]), _clean(bounds[key][1])],
        }
        for key in sa
        if key != "n"
    ]
    return {
        "a": sources[0],
        "b": sources[1],
        "n_items": len(a),
        "matched_by": "id" if a.ids is not None and b.ids is not None else "position",
        "binning": {"strategy": strategy, "requested_bins": int(n_bins)},
        "coverage": coverage,
        "intervals": {
            "level": level,
            "n_resamples": n_resamples,
            "seed": seed,
            "n_groups": None if a.group is None else len(set(a.group)),
        },
        "rows": rows,
    }


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, allow_nan=False) + "\n"


def _number(value: float | None) -> str:
    if value is None:
        return "-"
    text = f"{value:.4f}"
    return "0.0000" if text == "-0.0000" else text


def _interval(bounds: Sequence[float | None]) -> str:
    return f"[{_number(bounds[0])}, {_number(bounds[1])}]"


def _resampling_note(intervals: dict[str, Any]) -> str:
    unit = "items" if intervals["n_groups"] is None else f"{intervals['n_groups']} groups"
    return (
        f"{intervals['level']:.0%} intervals: percentile bootstrap over {unit}, "
        f"{intervals['n_resamples']} resamples."
    )


def render_markdown(report: dict[str, Any]) -> str:
    """Markdown for the report of ``calibcheck metrics``."""
    binning = report["binning"]
    intervals = report.get("intervals")
    coverage = report.get("coverage", DEFAULT_COVERAGE)
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
    ]
    if report.get("top_label"):
        lines.append("Confidence is the largest probability of each vector (top-label).")
    if intervals:
        lines.append(_resampling_note(intervals))
        lines += ["", f"| metric | value | {intervals['level']:.0%} interval |", "|---|---|---|"]
    else:
        lines += ["", "| metric | value |", "|---|---|"]
    for key, _ in METRIC_LABELS:
        if key not in report["metrics"]:
            continue
        line = f"| {_label(key, coverage)} | {_number(report['metrics'][key])} |"
        if intervals:
            line += f" {_interval(intervals['bounds'][key])} |"
        lines.append(line)
    if binning["singleton_bins"]:
        lines += [
            "",
            f"{binning['singleton_bins']} bin(s) hold a single item and add nothing to the "
            "debiased squared error.",
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
    if report.get("risk_coverage"):
        lines += [
            "",
            "| coverage | "
            + " | ".join(f"{p['coverage']:.0%}" for p in report["risk_coverage"])
            + " |",
        ]
        lines.append("|---|" + "---|" * len(report["risk_coverage"]))
        lines.append(
            "| accuracy | "
            + " | ".join(_number(p["accuracy"]) for p in report["risk_coverage"])
            + " |"
        )
    stated = report.get("stated")
    if stated:
        lines += [
            "",
            "## Stated confidence",
            "",
            f"{stated['distinct_values']} distinct values; "
            f"{stated['share_multiple_of_10']:.1%} of answers are multiples of 10 % and "
            f"{stated['share_multiple_of_5']:.1%} are multiples of 5 %.",
            "",
            "| stated | items | share |",
            "|---|---|---|",
        ]
        for top in stated["top_values"]:
            lines.append(f"| {top['confidence']:.2%} | {top['count']} | {top['share']:.1%} |")
    return "\n".join(lines) + "\n"


def render_recalibration_markdown(report: dict[str, Any]) -> str:
    title = "# Recalibration report"
    if report.get("source"):
        title += f": {report['source']}"
    how = (
        "each map is fitted on the other file"
        if report["folds"] is None
        else f"each map is fitted on {report['folds'] - 1} of {report['folds']} folds and "
        "scored on the held-out fold" + (" (groups kept together)" if report["grouped"] else "")
    )
    binning = report["binning"]
    lines = [
        title,
        "",
        f"{report['n_items']} items; {how}. Metrics use {binning['requested_bins']} "
        f"{BINNING_NAMES[binning['strategy']]} bins.",
        "",
        "| method | fitted on | " + " | ".join(label for _, label in RECALIBRATION_COLUMNS) + " |",
        "|---|---|" + "---|" * len(RECALIBRATION_COLUMNS),
    ]
    for row in report["rows"]:
        cells = " | ".join(_number(row[key]) for key, _ in RECALIBRATION_COLUMNS)
        lines.append(f"| {row['method']} | {row['fit']} | {cells} |")
    for row in report["rows"]:
        if row.get("temperatures"):
            values = ", ".join(f"{t:.3f}" for t in row["temperatures"])
            lines += ["", f"Temperatures fitted on vectors: {values}."]
    return "\n".join(lines) + "\n"


def render_comparison_markdown(report: dict[str, Any]) -> str:
    intervals = report["intervals"]
    coverage = report["coverage"]
    lines = [
        "# Calibration comparison",
        "",
        f"A: {report['a']}",
        f"B: {report['b']}",
        "",
        f"{report['n_items']} items matched by {report['matched_by']}. Differences are A - B; "
        + _resampling_note(intervals).replace("percentile bootstrap", "paired bootstrap"),
        "",
        f"| metric | A | B | A - B | {intervals['level']:.0%} interval |",
        "|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {_label(row['metric'], coverage)} | {_number(row['a'])} | {_number(row['b'])} "
            f"| {_number(row['difference'])} | {_interval(row['interval'])} |"
        )
    return "\n".join(lines) + "\n"
