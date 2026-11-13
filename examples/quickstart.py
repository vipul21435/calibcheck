"""Use calibcheck from Python on the bundled simulated example.

Run from the repository root:

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from calibcheck import (
    bootstrap,
    consistency_resampling,
    cross_fit,
    digit_preference,
    read_items,
    summary,
)

items = read_items(Path(__file__).with_name("example.jsonl"))
conf, correct = items.confidence, items.correct

metrics = summary(conf, correct, strategy="mass")
intervals = bootstrap(
    lambda idx: summary(conf[idx], correct[idx], strategy="mass"), len(items), seed=0
)
for key in ("accuracy", "ece", "ece_debiased", "l2_ce_debiased", "brier", "auroc"):
    lo, hi = intervals[key]
    print(f"{key:>15}: {metrics[key]:.4f}  [{lo:.4f}, {hi:.4f}]")

test = consistency_resampling(conf, correct, strategy="mass", seed=0)
print(
    f"consistency test: ECE {test.ece:.4f}, under calibration {test.null_mean:.4f} on average, "
    f"p = {test.p_value:.4f}"
)

for method in ("temperature", "platt", "isotonic", "histogram"):
    after = summary(cross_fit(method, conf, correct, seed=0), correct, strategy="mass")
    print(
        f"{method:>15}: debiased ECE {metrics['ece_debiased']:.4f} -> {after['ece_debiased']:.4f}"
    )

profile = digit_preference(conf, top=3)
top = ", ".join(f"{v['confidence']:.0%} ({v['share']:.0%})" for v in profile["top_values"])
print(f"{profile['distinct_values']} distinct confidences; most frequent: {top}")
