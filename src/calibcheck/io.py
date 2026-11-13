"""Read items from JSONL or CSV: a confidence, a correctness flag, an optional group."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TRUE_WORDS = frozenset({"1", "true", "t", "yes", "y", "correct"})
FALSE_WORDS = frozenset({"0", "false", "f", "no", "n", "incorrect"})


@dataclass(frozen=True)
class Items:
    """Columns of one input file, in file order."""

    confidence: np.ndarray
    correct: np.ndarray
    group: list[str] | None = None

    def __len__(self) -> int:
        return int(self.confidence.size)


def parse_confidence(value: Any) -> float:
    """A number in [0, 1], given as a JSON number or a numeric string."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"confidence must be a number, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"confidence must be a number, got {value!r}") from None
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise ValueError(f"confidence must be a number, got {value!r}")
    if math.isnan(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"confidence must lie in [0, 1], got {value!r}")
    return number


def parse_correct(value: Any) -> float:
    """1.0 or 0.0 from a boolean, 0/1, or a word such as "true", "no", "correct"."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in TRUE_WORDS:
            return 1.0
        if text in FALSE_WORDS:
            return 0.0
        try:
            number = float(text)
        except ValueError:
            number = math.nan
        if number in (0.0, 1.0):
            return number
    raise ValueError(f"correct must be a boolean or 0/1, got {value!r}")


def _jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from None
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object per line")
            yield lineno, row


def _csv_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield reader.line_num, row


def read_items(
    path: str | Path,
    *,
    confidence: str = "confidence",
    correct: str = "correct",
    group: str | None = None,
) -> Items:
    """Load a ``.jsonl`` / ``.ndjson`` or ``.csv`` file into arrays.

    ``confidence``, ``correct`` and ``group`` name the fields (JSON keys or CSV columns).
    Every problem is reported with the file name and line number.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        rows = _jsonl_rows(path)
    elif suffix == ".csv":
        rows = _csv_rows(path)
    else:
        raise ValueError(f"{path}: unsupported file type {suffix!r}; expected .jsonl or .csv")

    conf: list[float] = []
    corr: list[float] = []
    groups: list[str] = []
    for lineno, row in rows:
        where = f"{path}:{lineno}"
        for name in (confidence, correct) + ((group,) if group else ()):
            if row.get(name) is None or row.get(name) == "":
                raise ValueError(f"{where}: missing field {name!r}")
        try:
            conf.append(parse_confidence(row[confidence]))
            corr.append(parse_correct(row[correct]))
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from None
        if group:
            groups.append(str(row[group]))
    if not conf:
        raise ValueError(f"{path}: no items")
    return Items(
        confidence=np.array(conf, dtype=float),
        correct=np.array(corr, dtype=float),
        group=groups if group else None,
    )
