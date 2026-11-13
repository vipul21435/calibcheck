"""Read items from JSONL or CSV.

Each item carries either a confidence and a correctness flag, or a vector of class
probabilities (or logits) and the index of the true class. A group and an id are optional.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .multiclass import pad_vectors, softmax, top_label
from .stated import parse_stated

TRUE_WORDS = frozenset({"1", "true", "t", "yes", "y", "correct"})
FALSE_WORDS = frozenset({"0", "false", "f", "no", "n", "incorrect"})
_SEPARATORS = re.compile(r"[;|,\s]+")


@dataclass(frozen=True)
class Items:
    """Columns of one input file, in file order.

    ``probs`` (rows sum to one) and ``logits`` are set when the file held vectors; rows
    of different lengths are padded (probability 0, logit ``-inf``) and ``ragged`` is true.
    """

    confidence: np.ndarray
    correct: np.ndarray
    group: list[str] | None = None
    ids: list[str] | None = None
    probs: np.ndarray | None = None
    logits: np.ndarray | None = None
    label: np.ndarray | None = None
    ragged: bool = False

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
    elif isinstance(value, float) or (isinstance(value, int) and abs(value) <= 1):
        number = float(value)
    elif isinstance(value, int):
        number = math.inf
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


def parse_vector(value: Any) -> list[float]:
    """A list of numbers: a JSON array, or text such as "[0.1, 0.9]" or "0.1;0.9"."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                raise ValueError(f"cannot read a vector from {value!r}") from None
        else:
            value = [part for part in _SEPARATORS.split(text) if part]
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"a vector needs at least two numbers, got {value!r}")
    try:
        numbers = [float(v) for v in value]
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"cannot read a vector from {str(value)[:80]!r}") from None
    if any(math.isnan(v) for v in numbers) or any(isinstance(v, bool) for v in value):
        raise ValueError(f"cannot read a vector from {value!r}")
    return numbers


def parse_label(value: Any, size: int) -> int:
    """Index of the true class, an integer in [0, size)."""
    try:
        number = float(value) if not isinstance(value, bool) else math.nan
    except (TypeError, ValueError, OverflowError):
        number = math.nan
    if math.isnan(number) or number != int(number) or not 0 <= int(number) < size:
        raise ValueError(f"label must be a class index in [0, {size}), got {value!r}")
    return int(number)


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
        try:
            for row in reader:
                yield reader.line_num, row
        except csv.Error as exc:
            raise ValueError(f"{path}:{reader.line_num}: invalid CSV: {exc}") from None


def read_items(
    path: str | Path,
    *,
    confidence: str = "confidence",
    correct: str = "correct",
    group: str | None = None,
    id_field: str | None = None,
    probs: str | None = None,
    logits: str | None = None,
    label: str = "label",
    stated: bool = False,
    words: Mapping[str, float] | None = None,
    scale: str = "auto",
) -> Items:
    """Load a ``.jsonl`` / ``.ndjson`` or ``.csv`` file into arrays.

    By default each row needs the fields named by ``confidence`` and ``correct``. With
    ``probs`` or ``logits`` (a field holding a vector) and ``label`` the confidence is the
    largest probability and the answer is correct when that class is the label. With
    ``stated=True`` the confidence may be text such as "85%" or a phrase from ``words``.
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
    if probs and logits:
        raise ValueError("give either a probability field or a logit field, not both")
    vector_field = probs or logits
    needed = [vector_field, label] if vector_field else [confidence, correct]
    needed += [name for name in (group, id_field) if name]

    conf: list[float] = []
    corr: list[float] = []
    vectors: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    ids: list[str] = []
    for lineno, row in rows:
        where = f"{path}:{lineno}"
        for name in needed:
            if row.get(name) is None or row.get(name) == "":
                raise ValueError(f"{where}: missing field {name!r}")
        try:
            if vector_field:
                vector = parse_vector(row[vector_field])
                if probs and (min(vector) < 0 or abs(sum(vector) - 1.0) > 1e-6):
                    raise ValueError("probabilities must be non-negative and sum to 1")
                vectors.append(vector)
                labels.append(parse_label(row[label], len(vector)))
            else:
                raw = row[confidence]
                conf.append(
                    parse_stated(raw, words=words, scale=scale) if stated else parse_confidence(raw)
                )
                corr.append(parse_correct(row[correct]))
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from None
        if group:
            groups.append(str(row[group]))
        if id_field:
            ids.append(str(row[id_field]))
    if not conf and not vectors:
        raise ValueError(f"{path}: no items")
    if id_field and len(set(ids)) != len(ids):
        raise ValueError(f"{path}: values of {id_field!r} are not unique")

    extra: dict[str, Any] = {}
    if vector_field:
        if logits:
            z, ragged = pad_vectors(vectors, -math.inf)
            p = softmax(z)
        else:
            p, ragged = pad_vectors(vectors, 0.0)
            with np.errstate(divide="ignore"):
                z = np.log(p)
        y = np.array(labels, dtype=int)
        confidence_values, correct_values = top_label(p, y)
        extra = {"probs": p, "logits": z, "label": y, "ragged": ragged}
    else:
        confidence_values = np.array(conf, dtype=float)
        correct_values = np.array(corr, dtype=float)
    return Items(
        confidence=confidence_values,
        correct=correct_values,
        group=groups if group else None,
        ids=ids if id_field else None,
        **extra,
    )
