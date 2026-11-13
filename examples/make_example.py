"""Write examples/example.jsonl: simulated answers of an overconfident model.

The data is synthetic, not measured from any model. Each item states a confidence taken
from a short list of round values and is correct with probability ``0.3 + 0.55 * c``:
slightly underconfident at 0.5 and 0.6, increasingly overconfident above that.

A second file, examples/example_b.jsonl, holds a second simulated model on the same items
that states that probability itself, so it is calibrated by construction. The two files
are the input of the ``compare`` example.

Run from the repository root:

    python examples/make_example.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
SHARES = [0.05, 0.08, 0.12, 0.25, 0.25, 0.15, 0.10]
N_ITEMS = 400
SEED = 2026


def main() -> None:
    rng = np.random.default_rng(SEED)
    confidence = rng.choice(LEVELS, size=N_ITEMS, p=SHARES)
    correct = rng.random(N_ITEMS) < 0.3 + 0.55 * confidence
    out = Path(__file__).with_name("example.jsonl")
    with out.open("w", encoding="utf-8") as handle:
        for i, (c, y) in enumerate(zip(confidence, correct, strict=True), start=1):
            row = {"id": f"q{i:03d}", "confidence": float(c), "correct": bool(y)}
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {N_ITEMS} items to {out.name}")

    # The generating process is known, so its calibration error is too.
    gaps = [c - (0.3 + 0.55 * c) for c in LEVELS]
    true_ece = sum(s * abs(g) for s, g in zip(SHARES, gaps, strict=True))
    true_sq = sum(s * g * g for s, g in zip(SHARES, gaps, strict=True))
    print(f"generating process: ECE {true_ece:.4f}, squared calibration error {true_sq:.4f}")

    rng_b = np.random.default_rng(SEED + 1)
    confidence_b = np.round(0.3 + 0.55 * confidence, 3)
    correct_b = rng_b.random(N_ITEMS) < confidence_b
    out_b = Path(__file__).with_name("example_b.jsonl")
    with out_b.open("w", encoding="utf-8") as handle:
        for i, (c, y) in enumerate(zip(confidence_b, correct_b, strict=True), start=1):
            row = {"id": f"q{i:03d}", "confidence": float(c), "correct": bool(y)}
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {N_ITEMS} items to {out_b.name}")


if __name__ == "__main__":
    main()
