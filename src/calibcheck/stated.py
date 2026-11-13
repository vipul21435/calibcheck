"""Confidence as a model states it in text: "85%", "0.85", or a word such as "high".

Work on confidence stated by language models (Kadavath et al., 2022, arXiv:2207.05221;
Tian et al., 2023, arXiv:2305.14975) elicits a number or a phrase in the answer itself.
Such confidences come from a small set of round values, which this module reports.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

# A neutral five-step scale. It is a convention, not a measurement: pass the map that
# matches the wording of your own prompt.
DEFAULT_WORDS: dict[str, float] = {
    "very low": 0.1,
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
    "very high": 0.9,
}

SCALES = ("auto", "unit", "percent")
_PERCENT = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*%$")
_NUMBER = re.compile(r"^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)$")


def _from_number(number: float, scale: str, shown: Any) -> float:
    if math.isnan(number) or number < 0:
        raise ValueError(f"cannot read a confidence from {shown!r}")
    if scale == "unit" or (scale == "auto" and number <= 1.0):
        value = number
    elif scale == "percent" or (scale == "auto" and number <= 100.0):
        value = number / 100.0
    else:
        raise ValueError(f"confidence {shown!r} is above 100")
    if value > 1.0:
        raise ValueError(f"confidence {shown!r} is outside the {scale} scale")
    return value


def parse_stated(
    value: Any, *, words: Mapping[str, float] | None = None, scale: str = "auto"
) -> float:
    """Turn a stated confidence into a probability in [0, 1].

    Accepted: a number or numeric text ("0.85", 0.85), a percentage ("85%", "85 %") and
    any phrase in ``words`` (case and surrounding punctuation ignored). A bare number
    follows ``scale``: "unit" reads it as a probability, "percent" divides it by 100, and
    "auto" (the default) reads values up to 1 as probabilities and values above 1 as
    percentages, so a bare 1 means certainty, not one percent.
    """
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; expected one of {SCALES}")
    if isinstance(value, bool) or value is None:
        raise ValueError(f"cannot read a confidence from {value!r}")
    if isinstance(value, int) and abs(value) > 100:
        raise ValueError(f"confidence {str(value)[:40]!r} is above 100")
    if isinstance(value, (int, float)):
        return _from_number(float(value), scale, value)
    if not isinstance(value, str):
        raise ValueError(f"cannot read a confidence from {value!r}")
    text = value.strip().strip("\"'").strip().lower().rstrip(".!").strip()
    percent = _PERCENT.match(text)
    if percent:
        return _from_number(float(percent.group(1)), "percent", value)
    if _NUMBER.match(text):
        return _from_number(float(text), scale, value)
    table = DEFAULT_WORDS if words is None else words
    lookup = {" ".join(k.lower().split()): float(v) for k, v in table.items()}
    key = " ".join(text.split())
    if key in lookup:
        number = lookup[key]
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"word scale maps {key!r} to {number}, outside [0, 1]")
        return number
    raise ValueError(f"cannot read a confidence from {value!r}")


def digit_preference(confidence: Iterable[float], *, top: int = 5) -> dict[str, Any]:
    """How a set of stated confidences clusters on round values.

    Returns the number of distinct values, the ``top`` most frequent values with their
    shares, and the shares of answers that are whole multiples of 10 % and of 5 %.
    Values are compared after rounding to a hundredth of a percent.
    """
    percent = [round(float(c) * 100.0, 2) for c in confidence]
    if not percent:
        raise ValueError("no items")
    n = len(percent)
    counts = Counter(percent)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))

    def share_of_multiples(step: float) -> float:
        hits = sum(1 for p in percent if abs(p / step - round(p / step)) < 1e-9)
        return hits / n

    return {
        "n_items": n,
        "distinct_values": len(counts),
        "top_values": [
            {"confidence": value / 100.0, "count": count, "share": count / n}
            for value, count in ordered[:top]
        ],
        "share_multiple_of_10": share_of_multiples(10.0),
        "share_multiple_of_5": share_of_multiples(5.0),
    }
