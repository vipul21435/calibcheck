"""Command line entry point: ``calibcheck metrics | diagram | recalibrate | compare``."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .binning import STRATEGIES
from .io import Items, read_items
from .metrics import DEFAULT_BINS, DEFAULT_COVERAGE
from .recalibrate import METHODS
from .report import (
    build_comparison_report,
    build_recalibration_report,
    build_report,
    render_comparison_markdown,
    render_json,
    render_markdown,
    render_recalibration_markdown,
)
from .stated import SCALES


def _load(args: argparse.Namespace, path: str, id_field: str | None = None) -> Items:
    words = None
    if args.words:
        words = json.loads(Path(args.words).read_text(encoding="utf-8"))
        if not isinstance(words, dict):
            raise ValueError(f"{args.words}: expected a JSON object mapping phrases to numbers")
    return read_items(
        path,
        confidence=args.confidence,
        correct=args.correct,
        group=args.group,
        id_field=id_field,
        probs=args.probs,
        logits=args.logits,
        label=args.label,
        stated=args.stated or bool(args.words),
        words=words,
        scale=args.scale,
    )


def _emit(args: argparse.Namespace, text: str) -> int:
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    items = _load(args, args.file)
    report = build_report(
        items.confidence,
        items.correct,
        n_bins=args.bins,
        strategy=args.binning,
        source=args.file,
        coverage=args.coverage,
        n_resamples=args.bootstrap,
        level=args.level,
        seed=args.seed,
        groups=items.group,
        probs=items.probs,
        labels=items.label,
        classwise=not items.ragged,
        stated=args.stated or bool(args.words),
    )
    return _emit(args, render_json(report) if args.format == "json" else render_markdown(report))


def _cmd_diagram(args: argparse.Namespace) -> int:
    from .diagram import reliability_diagram

    items = _load(args, args.file)
    result = reliability_diagram(
        items.confidence,
        items.correct,
        args.output,
        n_bins=args.bins,
        strategy=args.binning,
        n_resamples=args.resamples,
        level=args.level,
        seed=args.seed,
        title=args.title,
    )
    print(f"wrote {args.output}")
    print(
        f"consistency test: plug-in ECE {result.ece:.4f}; under calibration its mean is "
        f"{result.null_mean:.4f} and its {result.level:.0%} quantile {result.null_quantile:.4f}; "
        f"p = {result.p_value:.4f} ({result.n_resamples} resamples)"
    )
    return 0


def _cmd_recalibrate(args: argparse.Namespace) -> int:
    items = _load(args, args.file)
    fit_items = _load(args, args.fit_on) if args.fit_on else None
    report = build_recalibration_report(
        items,
        methods=tuple(METHODS) if args.method == "all" else (args.method,),
        n_folds=args.folds,
        seed=args.seed,
        n_bins=args.bins,
        strategy=args.binning,
        fit_items=fit_items,
        in_sample=args.in_sample,
        source=args.file,
    )
    text = render_json(report) if args.format == "json" else render_recalibration_markdown(report)
    return _emit(args, text)


def _cmd_compare(args: argparse.Namespace) -> int:
    a = _load(args, args.file_a, id_field=args.id)
    b = _load(args, args.file_b, id_field=args.id)
    report = build_comparison_report(
        a,
        b,
        n_bins=args.bins,
        strategy=args.binning,
        coverage=args.coverage,
        n_resamples=args.bootstrap,
        level=args.level,
        seed=args.seed,
        sources=(args.file_a, args.file_b),
    )
    text = render_json(report) if args.format == "json" else render_comparison_markdown(report)
    return _emit(args, text)


def _add_input_options(parser: argparse.ArgumentParser) -> None:
    fields = parser.add_argument_group("input fields")
    fields.add_argument("--confidence", default="confidence", metavar="FIELD")
    fields.add_argument("--correct", default="correct", metavar="FIELD")
    fields.add_argument("--probs", metavar="FIELD", help="field with a probability vector")
    fields.add_argument("--logits", metavar="FIELD", help="field with a vector of logits")
    fields.add_argument("--label", default="label", metavar="FIELD", help="true class index")
    fields.add_argument("--group", metavar="FIELD", help="resample and split by this field")
    fields.add_argument(
        "--stated", action="store_true", help='read confidences such as "85%%" or "high"'
    )
    fields.add_argument("--words", metavar="JSON", help="file mapping phrases to probabilities")
    fields.add_argument(
        "--scale", choices=SCALES, default="auto", help="how to read a bare stated number"
    )
    binning = parser.add_argument_group("binning")
    binning.add_argument("--bins", type=int, default=DEFAULT_BINS, metavar="N")
    binning.add_argument("--binning", choices=STRATEGIES, default="width")
    parser.add_argument("--seed", type=int, default=0, help="seed of every random draw")
    parser.add_argument("--level", type=float, default=0.95, help="interval level")


def _add_report_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("md", "json"), default="md")
    parser.add_argument("--output", "-o", metavar="PATH", help="write the report to a file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibcheck",
        description="Measure how well stated confidence matches observed accuracy.",
    )
    parser.add_argument("--version", action="version", version=f"calibcheck {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    metrics = commands.add_parser(
        "metrics",
        help="calibration metrics with bootstrap intervals for one file",
        description="Read a .jsonl or .csv file with a confidence in [0, 1] and a 0/1 "
        "correctness per item (or a probability vector and a label) and report metrics.",
    )
    metrics.add_argument("file", help="input file (.jsonl, .ndjson or .csv)")
    _add_input_options(metrics)
    metrics.add_argument(
        "--bootstrap", type=int, default=1000, metavar="N", help="resamples (0 for none)"
    )
    metrics.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    _add_report_options(metrics)
    metrics.set_defaults(func=_cmd_metrics)

    diagram = commands.add_parser(
        "diagram",
        help="reliability diagram with consistency bands (PNG, needs matplotlib)",
    )
    diagram.add_argument("file")
    _add_input_options(diagram)
    diagram.add_argument("--output", "-o", required=True, metavar="PNG")
    diagram.add_argument("--resamples", type=int, default=1000, metavar="N")
    diagram.add_argument("--title")
    diagram.set_defaults(func=_cmd_diagram)

    recalibrate = commands.add_parser(
        "recalibrate",
        help="metrics before and after recalibration fitted on held-out items",
    )
    recalibrate.add_argument("file")
    _add_input_options(recalibrate)
    recalibrate.add_argument("--method", choices=("all", *METHODS), default="all")
    recalibrate.add_argument("--folds", type=int, default=5, metavar="K")
    recalibrate.add_argument(
        "--fit-on", metavar="FILE", help="fit on this file instead of cross-fitting"
    )
    recalibrate.add_argument(
        "--in-sample",
        action="store_true",
        help="also show each map fitted and scored on the same items",
    )
    _add_report_options(recalibrate)
    recalibrate.set_defaults(func=_cmd_recalibrate)

    compare = commands.add_parser(
        "compare", help="two models on the same items, paired bootstrap of the difference"
    )
    compare.add_argument("file_a")
    compare.add_argument("file_b")
    _add_input_options(compare)
    compare.add_argument("--id", metavar="FIELD", help="match items by this field")
    compare.add_argument("--bootstrap", type=int, default=1000, metavar="N")
    compare.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    _add_report_options(compare)
    compare.set_defaults(func=_cmd_compare)
    return parser


def _show_warning(message, category, filename, lineno, file=None, line=None) -> None:
    print(f"calibcheck: warning: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with warnings.catch_warnings():
        warnings.showwarning = _show_warning
        try:
            return args.func(args)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"calibcheck: error: {exc}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
