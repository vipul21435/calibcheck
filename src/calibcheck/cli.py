"""Command line entry point: ``calibcheck metrics FILE``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .binning import STRATEGIES
from .io import read_items
from .metrics import DEFAULT_BINS
from .report import build_report, render_json, render_markdown


def _cmd_metrics(args: argparse.Namespace) -> int:
    items = read_items(args.file, confidence=args.confidence, correct=args.correct)
    report = build_report(
        items.confidence,
        items.correct,
        n_bins=args.bins,
        strategy=args.binning,
        source=args.file,
    )
    text = render_json(report) if args.format == "json" else render_markdown(report)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibcheck",
        description="Measure how well stated confidence matches observed accuracy.",
    )
    parser.add_argument("--version", action="version", version=f"calibcheck {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    metrics = commands.add_parser(
        "metrics",
        help="calibration metrics and the per-bin table for one file",
        description="Read a .jsonl or .csv file with a confidence in [0, 1] and a 0/1 "
        "correctness per item, and report calibration metrics.",
    )
    metrics.add_argument("file", help="input file (.jsonl, .ndjson or .csv)")
    metrics.add_argument(
        "--confidence", default="confidence", metavar="FIELD", help="confidence field name"
    )
    metrics.add_argument(
        "--correct", default="correct", metavar="FIELD", help="correctness field name"
    )
    metrics.add_argument(
        "--bins", type=int, default=DEFAULT_BINS, metavar="N", help="number of bins (default 15)"
    )
    metrics.add_argument(
        "--binning",
        choices=STRATEGIES,
        default="width",
        help="equal-width bins, or equal-mass bins with ties kept together (default width)",
    )
    metrics.add_argument(
        "--format", choices=("md", "json"), default="md", help="report format (default md)"
    )
    metrics.add_argument("--output", "-o", metavar="PATH", help="write the report to a file")
    metrics.set_defaults(func=_cmd_metrics)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError) as exc:
        print(f"calibcheck: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
