"""Bias of the plug-in and the debiased estimators on simulated data with a known truth.

A model states one of eight confidences (0.25, 0.35, ..., 0.95) and is right with that
probability minus a shift, so the true ECE is the shift and the true squared calibration
error its square. Each row averages 3000 samples of 200 items, with ten equal-width bins.

Run from the repository root:

    python examples/bias_simulation.py
"""

from __future__ import annotations

import numpy as np

from calibcheck import bin_stats

LEVELS = np.arange(2, 10) / 10 + 0.05
N_ITEMS, N_SAMPLES, SEED = 200, 3000, 0


def main() -> None:
    print("true ECE | plug-in ECE | debiased ECE | true sq. | plug-in sq. | debiased sq.")
    for shift in (0.0, 0.05, 0.1):
        rng = np.random.default_rng(SEED)
        rows = []
        for _ in range(N_SAMPLES):
            conf = rng.choice(LEVELS, size=N_ITEMS)
            correct = rng.random(N_ITEMS) < conf - shift
            stats = bin_stats(conf, correct, n_bins=10)
            rows.append(
                [stats.ece(), stats.ece(debias=True), stats.squared_ce(), stats.squared_ce(True)]
            )
        ece, ece_db, sq, sq_db = np.mean(rows, axis=0)
        print(
            f"{shift:8.4f} | {ece:11.4f} | {ece_db:12.4f} | {shift**2:8.4f} | {sq:11.4f} "
            f"| {sq_db:12.4f}"
        )


if __name__ == "__main__":
    main()
