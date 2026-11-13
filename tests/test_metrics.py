import math

import numpy as np
import pytest

from calibcheck.binning import assign_bins
from calibcheck.metrics import (
    bin_stats,
    brier_score,
    ece,
    mce,
    murphy_decomposition,
    squared_calibration_error,
    summary,
)

# Six items in two of ten equal-width bins, worked by hand:
#   [0.9, 1.0]: 4 items, conf 0.90, acc 0.75, gap -0.15
#   [0.6, 0.7): 2 items, conf 0.60, acc 0.50, gap -0.10
CONF = [0.9, 0.9, 0.9, 0.9, 0.6, 0.6]
CORRECT = [1, 1, 1, 0, 1, 0]


def exactly_calibrated():
    """Levels k/10 with exactly k of 10 items correct at each level."""
    conf, correct = [], []
    for k in range(11):
        conf += [k / 10] * 10
        correct += [1] * k + [0] * (10 - k)
    return np.array(conf), np.array(correct)


def test_hand_worked_ece_and_mce():
    assert ece(CONF, CORRECT, n_bins=10) == pytest.approx(4 / 6 * 0.15 + 2 / 6 * 0.10, abs=1e-12)
    assert mce(CONF, CORRECT, n_bins=10) == pytest.approx(0.15, abs=1e-12)


def test_hand_worked_squared_calibration_error():
    plugin = squared_calibration_error(CONF, CORRECT, n_bins=10, debias=False)
    assert plugin == pytest.approx(4 / 6 * 0.0225 + 2 / 6 * 0.01, abs=1e-12)
    # debiased: 0.0225 - 0.75 * 0.25 / 3 = -0.04 and 0.01 - 0.5 * 0.5 / 1 = -0.24
    debiased = squared_calibration_error(CONF, CORRECT, n_bins=10)
    assert debiased == pytest.approx(4 / 6 * -0.04 + 2 / 6 * -0.24, abs=1e-12)


def test_bin_table_of_the_hand_worked_example():
    stats = bin_stats(CONF, CORRECT, n_bins=10)
    assert stats.n_bins == 10
    assert stats.n_items == 6
    assert stats.count.tolist() == [0, 0, 0, 0, 0, 0, 2, 0, 0, 4]
    assert stats.confidence[9] == pytest.approx(0.9)
    assert stats.accuracy[9] == pytest.approx(0.75)
    assert stats.gap[6] == pytest.approx(-0.10)
    assert np.isnan(stats.confidence[0]) and np.isnan(stats.accuracy[0])


@pytest.mark.parametrize("strategy", ["width", "mass"])
def test_exactly_calibrated_data_has_zero_error(strategy):
    conf, correct = exactly_calibrated()
    out = summary(conf, correct, n_bins=20, strategy=strategy)
    assert out["ece"] == pytest.approx(0.0, abs=1e-12)
    assert out["mce"] == pytest.approx(0.0, abs=1e-12)
    assert out["sq_ce_plugin"] == pytest.approx(0.0, abs=1e-12)
    assert out["reliability"] == pytest.approx(0.0, abs=1e-12)
    assert out["accuracy"] == pytest.approx(out["mean_confidence"], abs=1e-12)
    # The debiased estimate goes below zero here, as an unbiased estimate must.
    assert out["sq_ce_debiased"] < 0.0
    assert out["l2_ce_debiased"] == 0.0


def test_empty_bins_carry_no_weight():
    conf = [0.05, 0.05, 0.95, 0.95]
    correct = [0, 1, 1, 1]
    stats = bin_stats(conf, correct, n_bins=10)
    assert int(np.sum(stats.count == 0)) == 8
    assert stats.ece() == pytest.approx(0.5 * 0.45 + 0.5 * 0.05, abs=1e-12)
    assert stats.mce() == pytest.approx(0.45, abs=1e-12)
    # More bins only add empty ones here, so nothing changes.
    assert ece(conf, correct, n_bins=100) == pytest.approx(stats.ece(), abs=1e-12)


def test_metrics_do_not_depend_on_item_order():
    rng = np.random.default_rng(3)
    conf = np.round(rng.random(400), 2)
    correct = rng.random(400) < conf
    order = rng.permutation(400)
    for strategy in ("width", "mass"):
        a = summary(conf, correct, n_bins=12, strategy=strategy)
        b = summary(conf[order], correct[order], n_bins=12, strategy=strategy)
        for key in a:
            assert a[key] == pytest.approx(b[key], abs=1e-12), key


def test_bin_with_one_item_adds_nothing_to_the_debiased_estimate():
    conf = [0.15, 0.85, 0.85, 0.85, 0.85]
    correct = [1, 1, 1, 0, 1]
    stats = bin_stats(conf, correct, n_bins=10)
    assert stats.singleton_bins == 1
    only_big_bin = 4 / 5 * ((0.85 - 0.75) ** 2 - 0.75 * 0.25 / 3)
    assert stats.squared_ce(debias=True) == pytest.approx(only_big_bin, abs=1e-12)
    with_singleton = 1 / 5 * 0.85**2 + 4 / 5 * (0.85 - 0.75) ** 2
    assert stats.squared_ce(debias=False) == pytest.approx(with_singleton, abs=1e-12)


def test_brier_score_by_hand():
    assert brier_score([1.0, 0.0, 0.5], [1, 0, 1]) == pytest.approx(0.25 / 3)
    assert brier_score([0.8, 0.8], [True, False]) == pytest.approx((0.04 + 0.64) / 2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_murphy_identity_is_exact_for_forecasts_on_a_few_levels(seed):
    rng = np.random.default_rng(seed)
    levels = (np.arange(10) + 0.5) / 10  # one level per equal-width bin
    conf = rng.choice(levels, size=997)
    correct = rng.random(997) < conf**1.5
    dec = murphy_decomposition(conf, correct, n_bins=10)
    assert dec.brier == pytest.approx(brier_score(conf, correct), abs=1e-15)
    rebuilt = dec.reliability - dec.resolution + dec.uncertainty
    assert abs(dec.brier - rebuilt) < 1e-12
    assert abs(dec.within_bin) < 1e-12


@pytest.mark.parametrize("strategy", ["width", "mass"])
def test_murphy_identity_after_replacing_forecasts_by_bin_means(strategy):
    rng = np.random.default_rng(4)
    conf = rng.beta(4, 2, size=1500)
    correct = rng.random(1500) < conf
    stats = bin_stats(conf, correct, n_bins=15, strategy=strategy)
    binned = stats.confidence[assign_bins(conf, stats.edges)]
    assert bin_stats(binned, correct, n_bins=15, strategy=strategy).count.tolist() == (
        stats.count.tolist()
    )
    dec = murphy_decomposition(binned, correct, n_bins=15, strategy=strategy)
    assert abs(dec.brier - (dec.reliability - dec.resolution + dec.uncertainty)) < 1e-12
    # Binning the forecasts leaves the three terms unchanged.
    raw = murphy_decomposition(conf, correct, n_bins=15, strategy=strategy)
    assert raw.reliability == pytest.approx(dec.reliability, abs=1e-12)
    assert raw.resolution == pytest.approx(dec.resolution, abs=1e-12)
    assert raw.uncertainty == pytest.approx(dec.uncertainty, abs=1e-12)
    assert abs(raw.within_bin) > 1e-6


def test_within_bin_term_matches_its_own_formula():
    rng = np.random.default_rng(5)
    conf = rng.random(800)
    correct = (rng.random(800) < conf).astype(float)
    stats = bin_stats(conf, correct, n_bins=8)
    idx = assign_bins(conf, stats.edges)
    dc = conf - stats.confidence[idx]
    dy = correct - stats.accuracy[idx]
    expected = float(np.mean(dc**2) - 2.0 * np.mean(dc * dy))
    dec = murphy_decomposition(conf, correct, n_bins=8)
    assert dec.within_bin == pytest.approx(expected, abs=1e-12)
    assert abs(dec.within_bin) > 1e-6


def test_reliability_is_the_plugin_squared_calibration_error():
    rng = np.random.default_rng(6)
    conf = rng.random(300)
    correct = rng.random(300) < 0.6
    out = summary(conf, correct, n_bins=10)
    assert out["reliability"] == pytest.approx(out["sq_ce_plugin"], abs=1e-15)
    assert out["l2_ce_plugin"] == pytest.approx(math.sqrt(out["sq_ce_plugin"]))
    assert out["uncertainty"] == pytest.approx(out["accuracy"] * (1 - out["accuracy"]))


def loop_metrics(conf, correct, edges):
    """The binned metrics written as plain loops, as a reference for the array code."""
    n = len(conf)
    members = [[] for _ in range(len(edges) - 1)]
    for c, y in zip(conf, correct, strict=True):
        b = 0
        for i in range(1, len(edges) - 1):
            if c >= edges[i]:
                b = i
        members[b].append((c, y))
    ece_, mce_, plugin, debiased = 0.0, 0.0, 0.0, 0.0
    for rows in members:
        if not rows:
            continue
        k = len(rows)
        conf_b = sum(c for c, _ in rows) / k
        acc_b = sum(y for _, y in rows) / k
        gap = abs(acc_b - conf_b)
        ece_ += k / n * gap
        mce_ = max(mce_, gap)
        plugin += k / n * gap**2
        if k >= 2:
            debiased += k / n * (gap**2 - acc_b * (1 - acc_b) / (k - 1))
    return ece_, mce_, plugin, debiased


@pytest.mark.parametrize("strategy", ["width", "mass"])
@pytest.mark.parametrize("seed", range(5))
def test_binned_metrics_match_the_plain_loops(strategy, seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(5, 400))
    conf = rng.random(n) if seed % 2 else np.round(rng.random(n), 1)
    correct = (rng.random(n) < 0.3 + 0.5 * conf).astype(float)
    n_bins = int(rng.integers(1, 30))
    stats = bin_stats(conf, correct, n_bins=n_bins, strategy=strategy)
    expected = loop_metrics(conf.tolist(), correct.tolist(), stats.edges.tolist())
    got = (stats.ece(), stats.mce(), stats.squared_ce(debias=False), stats.squared_ce(debias=True))
    assert got == pytest.approx(expected, abs=1e-12)


def simulate(true_shift: float, n: int, reps: int, seed: int) -> tuple[float, float, float]:
    """Mean plug-in and debiased squared error for a model that states ``s`` and is right
    with probability ``s - true_shift``; the true squared error is ``true_shift ** 2``."""
    rng = np.random.default_rng(seed)
    levels = np.arange(2, 10) / 10 + 0.05  # 0.25 ... 0.95, the centres of eight bins
    plugin, debiased = [], []
    for _ in range(reps):
        conf = rng.choice(levels, size=n)
        correct = rng.random(n) < conf - true_shift
        stats = bin_stats(conf, correct, n_bins=10)
        plugin.append(stats.squared_ce(debias=False))
        debiased.append(stats.squared_ce(debias=True))
    return float(np.mean(plugin)), float(np.mean(debiased)), true_shift**2


@pytest.mark.parametrize("true_shift", [0.0, 0.1])
def test_plugin_is_biased_upward_and_the_debiased_estimator_is_not(true_shift):
    plugin, debiased, truth = simulate(true_shift, n=200, reps=3000, seed=7)
    # Expected plug-in bias is about (number of bins) * mean variance / n = 8 * 0.2 / 200.
    assert plugin - truth > 0.005
    assert abs(debiased - truth) < 0.0006
    assert abs(debiased - truth) < 0.1 * (plugin - truth)


def test_plugin_bias_shrinks_with_more_items_per_bin():
    small, _, _ = simulate(0.0, n=100, reps=1500, seed=8)
    large, _, _ = simulate(0.0, n=800, reps=1500, seed=8)
    assert small > 4 * large > 0


@pytest.mark.parametrize(
    ("conf", "correct", "message"),
    [
        ([], [], "no items"),
        ([0.5, 0.6], [1], "same length"),
        ([0.5, 1.2], [1, 0], r"\[0, 1\]"),
        ([-0.1, 0.2], [1, 0], r"\[0, 1\]"),
        ([0.5, float("nan")], [1, 0], "NaN"),
        ([0.5, 0.6], [1, 2], "0/1"),
        ([0.5, 0.6], ["yes", "no"], "0/1"),
        ([[0.5, 0.6]], [[1, 0]], "one-dimensional"),
    ],
)
def test_invalid_input_is_rejected(conf, correct, message):
    with pytest.raises(ValueError, match=message):
        summary(conf, correct)


def test_booleans_and_integers_are_both_accepted():
    as_bool = summary([0.7, 0.2, 0.9], [True, False, True])
    as_int = summary([0.7, 0.2, 0.9], [1, 0, 1])
    assert as_bool == as_int
