"""False-alarm rate of the consistency test on calibrated data with continuous confidences.

Each run draws 200 confidences from Beta(5, 2) and outcomes that are right with exactly
that probability, then tests calibration at the 5 % level with 15 equal-width bins and 1000
surrogates. A valid test rejects about 5 % of such runs.

Run from the repository root (a few minutes):

    python examples/false_alarm_rate.py
"""

from __future__ import annotations

import numpy as np

from calibcheck import consistency_resampling

N_RUNS, N_ITEMS, N_SURROGATES = 2000, 200, 1000


def main() -> None:
    rejections = 0
    for seed in range(N_RUNS):
        rng = np.random.default_rng(10_000 + seed)
        conf = rng.beta(5, 2, size=N_ITEMS)
        correct = rng.random(N_ITEMS) < conf
        result = consistency_resampling(conf, correct, n_resamples=N_SURROGATES, seed=seed)
        rejections += result.p_value <= 0.05
    rate = rejections / N_RUNS
    error = (rate * (1 - rate) / N_RUNS) ** 0.5
    print(
        f"{rejections} of {N_RUNS} calibrated samples rejected at the 5 % level: "
        f"{rate:.4f} (standard error {error:.4f})"
    )


if __name__ == "__main__":
    main()
