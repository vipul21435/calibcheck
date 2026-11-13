import itertools
import math

import numpy as np
import pytest
from scipy import stats as scipy_stats
from sklearn import metrics as sk

from calibcheck.metrics import (
    accuracy_at_coverage,
    aurc,
    auroc,
    bin_stats,
    ece,
    folded_normal_mean,
    log_loss,
    risk_coverage,
    summary,
)


def test_folded_normal_mean_matches_scipy():
    mu = np.array([0.0, 0.05, 0.2, 0.7, 0.3])
    sd = np.array([0.1, 0.1, 0.05, 0.3, 0.0])
    expected = [
        scipy_stats.foldnorm(c=m / s, scale=s).mean() if s > 0 else m
        for m, s in zip(mu, sd, strict=True)
    ]
    assert folded_normal_mean(mu, sd) == pytest.approx(expected, abs=1e-12)
    assert folded_normal_mean(-mu, sd) == pytest.approx(expected, abs=1e-12)


def test_debiased_ece_equals_the_simulated_definition():
    # Definition G.3 of Kumar, Liang and Ma: redraw each bin accuracy from a normal.
    rng = np.random.default_rng(0)
    conf = np.round(rng.random(600), 1)
    correct = rng.random(600) < 0.2 + 0.6 * conf
    stats = bin_stats(conf, correct, n_bins=10)
    m = stats.nonempty
    w, acc, cnt = stats.weight[m], stats.accuracy[m], stats.count[m]
    draws = rng.normal(acc, np.sqrt(acc * (1 - acc) / cnt), size=(200_000, int(m.sum())))
    expected_plugin = np.mean(np.abs(draws - stats.confidence[m]) @ w)
    simulated = stats.ece() - (expected_plugin - stats.ece())
    assert stats.ece(debias=True) == pytest.approx(simulated, abs=3e-4)
    assert ece(conf, correct, n_bins=10, debias=True) == stats.ece(debias=True)
    assert stats.ece(debias=True) < stats.ece()


def test_bins_with_accuracy_zero_or_one_get_no_correction():
    conf = [0.95] * 20 + [0.05] * 20
    correct = [1] * 20 + [0] * 20
    stats = bin_stats(conf, correct, n_bins=10)
    assert stats.ece(debias=True) == pytest.approx(stats.ece(), abs=1e-15)
    assert stats.ece() == pytest.approx(0.05)


def simulate_ece(true_shift: float, n: int, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    levels = np.arange(2, 10) / 10 + 0.05
    plugin, debiased = [], []
    for _ in range(reps):
        conf = rng.choice(levels, size=n)
        correct = rng.random(n) < conf - true_shift
        stats = bin_stats(conf, correct, n_bins=10)
        plugin.append(stats.ece())
        debiased.append(stats.ece(debias=True))
    return float(np.mean(plugin)), float(np.mean(debiased))


@pytest.mark.parametrize(("true_shift", "max_share"), [(0.0, 0.7), (0.1, 0.8)])
def test_debiasing_reduces_the_bias_of_the_ece(true_shift, max_share):
    plugin, debiased = simulate_ece(true_shift, n=200, reps=3000, seed=11)
    plugin_bias = plugin - true_shift
    # With no true error the plug-in ECE is still about 0.07 here.
    assert plugin_bias > 0.01
    assert abs(debiased - true_shift) < max_share * plugin_bias


def test_log_loss_matches_sklearn_and_survives_certainty():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.02, 0.98, size=300)
    correct = rng.random(300) < conf
    assert log_loss(conf, correct) == pytest.approx(sk.log_loss(correct, conf), abs=1e-12)
    assert log_loss([1.0, 0.0], [1, 0]) == pytest.approx(0.0, abs=1e-11)
    assert log_loss([1.0], [0]) == pytest.approx(-math.log(1e-12))


@pytest.mark.parametrize("decimals", [1, 2, 8])
def test_auroc_matches_sklearn_with_and_without_ties(decimals):
    rng = np.random.default_rng(2)
    conf = np.round(rng.random(500), decimals)
    correct = rng.random(500) < 0.2 + 0.6 * conf
    assert auroc(conf, correct) == pytest.approx(sk.roc_auc_score(correct, conf), abs=1e-12)


def test_auroc_edge_cases():
    assert auroc([0.1, 0.9], [0, 1]) == 1.0
    assert auroc([0.9, 0.1], [0, 1]) == 0.0
    assert auroc([0.5, 0.5, 0.5], [0, 1, 1]) == 0.5
    assert math.isnan(auroc([0.2, 0.8], [1, 1]))
    assert summary([0.2, 0.8], [1, 1])["auroc"] != summary([0.2, 0.8], [1, 1])["auroc"]


def test_risk_coverage_without_ties_matches_a_loop():
    rng = np.random.default_rng(3)
    conf = rng.random(200)
    correct = (rng.random(200) < conf).astype(float)
    order = np.argsort(-conf)
    risks = [1.0 - correct[order][:k].mean() for k in range(1, 201)]
    curve = risk_coverage(conf, correct)
    assert curve.risk == pytest.approx(risks, abs=1e-12)
    assert curve.coverage[0] == 1 / 200 and curve.coverage[-1] == 1.0
    assert aurc(conf, correct) == pytest.approx(np.mean(risks), abs=1e-12)
    assert accuracy_at_coverage(conf, correct, 0.5) == pytest.approx(1.0 - risks[99])
    assert accuracy_at_coverage(conf, correct, 1.0) == pytest.approx(correct.mean())


def test_risk_coverage_with_ties_is_the_mean_over_all_orders():
    conf = np.array([0.9, 0.9, 0.9, 0.5, 0.5, 0.2])
    correct = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    total = np.zeros(6)
    count = 0
    for first in itertools.permutations(range(3)):
        for second in itertools.permutations(range(3, 5)):
            order = list(first) + list(second) + [5]
            errors = 1.0 - correct[order]
            total += np.cumsum(errors) / np.arange(1, 7)
            count += 1
    curve = risk_coverage(conf, correct)
    assert curve.risk == pytest.approx(total / count, abs=1e-12)
    shuffled = np.random.default_rng(4).permutation(6)
    assert risk_coverage(conf[shuffled], correct[shuffled]).risk == pytest.approx(curve.risk)


def test_aurc_limits():
    # One shared confidence: every coverage has the overall error rate.
    assert aurc([0.7] * 10, [1] * 6 + [0] * 4) == pytest.approx(0.4)
    # Errors ranked last: the best possible curve for 2 errors in 4 items.
    assert risk_coverage([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]).risk == pytest.approx(
        [0.0, 0.0, 1 / 3, 0.5]
    )
    with pytest.raises(ValueError, match="coverage"):
        accuracy_at_coverage([0.5, 0.6], [1, 0], 0.0)


def test_summary_reports_the_new_metrics():
    rng = np.random.default_rng(5)
    conf = rng.random(300)
    correct = rng.random(300) < conf
    out = summary(conf, correct, coverage=0.5)
    assert out["log_loss"] == pytest.approx(log_loss(conf, correct))
    assert out["auroc"] == pytest.approx(auroc(conf, correct))
    assert out["aurc"] == pytest.approx(aurc(conf, correct))
    assert out["accuracy_at_coverage"] == pytest.approx(accuracy_at_coverage(conf, correct, 0.5))
    assert out["ece_debiased"] == pytest.approx(ece(conf, correct, debias=True))
