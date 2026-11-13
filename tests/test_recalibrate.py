import math
import warnings

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression as SkIsotonic
from sklearn.linear_model import LogisticRegression

from calibcheck.metrics import brier_score, ece, log_loss
from calibcheck.multiclass import softmax
from calibcheck.recalibrate import (
    HistogramBinning,
    IsotonicRegression,
    PlattScaling,
    TemperatureScaling,
    apply_temperature,
    at_temperature_bound,
    cross_fit,
    cross_fit_temperature,
    fit_apply,
    fit_temperature,
    fold_ids,
    logit,
    make_calibrator,
    pool_adjacent_violators,
    sigmoid,
)


def overconfident(seed, n=4000, sharpen=2.5):
    """Confidences whose logit is ``sharpen`` times too large."""
    rng = np.random.default_rng(seed)
    score = rng.normal(1.0, 1.2, size=n)
    correct = (rng.random(n) < sigmoid(score)).astype(float)
    return sigmoid(sharpen * score), correct


@pytest.mark.parametrize("decimals", [2, 6])
def test_isotonic_regression_matches_sklearn(decimals):
    rng = np.random.default_rng(0)
    conf = np.round(rng.random(700), decimals)
    correct = (rng.random(700) < conf**2).astype(float)
    new = np.concatenate([rng.random(300), [0.0, 1.0, conf.min() / 2]])
    ours = IsotonicRegression().fit(conf, correct)
    theirs = SkIsotonic(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(conf, correct)
    assert ours.predict(new) == pytest.approx(theirs.predict(new), abs=1e-12)
    assert ours.predict(conf) == pytest.approx(theirs.predict(conf), abs=1e-12)
    assert np.all(np.diff(ours.predict(np.sort(new))) >= -1e-15)


def test_pool_adjacent_violators():
    assert pool_adjacent_violators([1.0, 2.0, 3.0]).tolist() == [1.0, 2.0, 3.0]
    assert pool_adjacent_violators([3.0, 1.0, 2.0]).tolist() == [2.0, 2.0, 2.0]
    assert pool_adjacent_violators([1.0, 3.0, 2.0, 4.0]).tolist() == [1.0, 2.5, 2.5, 4.0]
    weighted = pool_adjacent_violators([1.0, 0.0], [3.0, 1.0])
    assert weighted.tolist() == [0.75, 0.75]


def test_temperature_is_recovered_from_synthetic_logits():
    rng = np.random.default_rng(1)
    true_logits = rng.normal(0.0, 2.0, size=(20000, 5))
    labels = np.array([rng.choice(5, p=p) for p in softmax(true_logits)])
    for scale in (0.5, 1.0, 3.0):
        assert fit_temperature(true_logits * scale, labels) == pytest.approx(scale, rel=0.03)
    scaled = apply_temperature(true_logits * 3.0, 3.0)
    assert scaled == pytest.approx(softmax(true_logits), abs=1e-12)


def test_temperature_fit_accepts_padded_vectors():
    rng = np.random.default_rng(2)
    logits = rng.normal(0.0, 2.0, size=(3000, 4))
    logits[::2, 3] = -math.inf  # every other item has three options
    labels = np.array([rng.choice(4, p=p) for p in softmax(logits)])
    assert fit_temperature(logits * 2.0, labels) == pytest.approx(2.0, rel=0.08)
    with pytest.raises(ValueError, match="one class index per item"):
        fit_temperature(logits, labels[:-1])


def test_binary_temperature_and_platt_recover_the_distortion():
    conf, correct = overconfident(3, n=20000)
    temperature = TemperatureScaling().fit(conf, correct)
    assert temperature.temperature == pytest.approx(2.5, rel=0.06)
    platt = PlattScaling().fit(conf, correct)
    assert platt.a == pytest.approx(1 / 2.5, rel=0.06)
    assert platt.b == pytest.approx(0.0, abs=0.06)
    assert log_loss(platt.predict(conf), correct) <= log_loss(temperature.predict(conf), correct)


def test_platt_scaling_matches_unpenalised_logistic_regression():
    conf, correct = overconfident(4, n=3000)
    conf = np.clip(conf, 1e-6, 1 - 1e-6)
    ours = PlattScaling().fit(conf, correct)
    theirs = LogisticRegression(C=1e12, tol=1e-12, max_iter=5000).fit(
        logit(conf).reshape(-1, 1), correct
    )
    assert ours.a == pytest.approx(theirs.coef_[0, 0], rel=1e-4)
    assert ours.b == pytest.approx(theirs.intercept_[0], abs=1e-4)


def test_histogram_binning_returns_bin_accuracies():
    conf = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
    correct = np.array([0, 1, 1, 1, 1, 0])
    model = HistogramBinning(n_bins=10, strategy="width").fit(conf, correct)
    assert model.predict([0.15, 0.95, 0.9]) == pytest.approx([0.5, 0.75, 0.75])
    assert model.predict([0.55]) == pytest.approx([0.55])  # empty bin: its midpoint
    by_mass = HistogramBinning(n_bins=2).fit(conf, correct)
    assert by_mass.predict([0.0, 0.5, 1.0]) == pytest.approx([0.5, 0.5, 0.75])


def test_fold_ids_are_balanced_reproducible_and_keep_groups_together():
    folds = fold_ids(103, 5, seed=1)
    assert sorted(np.bincount(folds).tolist()) == [20, 20, 21, 21, 21]
    assert folds.tolist() == fold_ids(103, 5, seed=1).tolist()
    assert folds.tolist() != fold_ids(103, 5, seed=2).tolist()
    groups = [f"g{i % 12}" for i in range(120)]
    grouped = fold_ids(120, 4, seed=0, groups=groups)
    for g in set(groups):
        assert len({grouped[i] for i in range(120) if groups[i] == g}) == 1
    assert sorted(np.bincount(grouped).tolist()) == [30, 30, 30, 30]
    with pytest.raises(ValueError, match="at least 2"):
        fold_ids(10, 1)
    with pytest.raises(ValueError, match="cannot make"):
        fold_ids(10, 4, groups=["a", "b"] * 5)
    with pytest.raises(ValueError, match="one entry per item"):
        fold_ids(10, 2, groups=["a"])


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic", "histogram"])
def test_cross_fitting_repairs_an_overconfident_model(method):
    conf, correct = overconfident(5)
    before = ece(conf, correct, debias=True)
    after = ece(cross_fit(method, conf, correct, seed=0), correct, debias=True)
    assert before > 0.1
    assert after < 0.03


def test_scoring_on_the_fitting_items_flatters_isotonic_regression():
    rng = np.random.default_rng(6)
    conf = rng.random(300)
    correct = (rng.random(300) < 0.6).astype(float)  # confidence carries no information
    same = IsotonicRegression().fit(conf, correct).predict(conf)
    held_out = cross_fit("isotonic", conf, correct, n_folds=5, seed=0)
    base = correct.mean() * (1 - correct.mean())  # Brier score of the constant base rate
    assert brier_score(same, correct) < base - 0.002
    assert brier_score(held_out, correct) > base
    assert ece(same, correct) < 1e-12 < ece(held_out, correct)


def test_cross_fit_never_uses_the_held_out_fold():
    conf, correct = overconfident(7, n=500)
    folds = fold_ids(500, 5, seed=3)
    out = cross_fit("platt", conf, correct, n_folds=5, seed=3)
    for f in range(5):
        expected = fit_apply("platt", conf[folds != f], correct[folds != f], conf[folds == f])
        assert out[folds == f] == pytest.approx(expected, abs=1e-12)


def test_cross_fit_temperature_on_vectors():
    rng = np.random.default_rng(8)
    true_logits = rng.normal(0.0, 1.5, size=(5000, 4))
    labels = np.array([rng.choice(4, p=p) for p in softmax(true_logits)])
    probs, temperatures = cross_fit_temperature(true_logits * 2.0, labels, n_folds=4, seed=0)
    assert len(temperatures) == 4
    assert temperatures == pytest.approx([2.0] * 4, rel=0.08)
    assert probs.sum(axis=1) == pytest.approx(np.ones(5000))
    assert np.argmax(probs, axis=1).tolist() == np.argmax(true_logits, axis=1).tolist()


def test_helpers():
    assert sigmoid([-800.0, 0.0, 800.0]).tolist() == [0.0, 0.5, 1.0]
    assert logit([0.5])[0] == 0.0
    assert math.isfinite(logit([0.0])[0]) and math.isfinite(logit([1.0])[0])
    assert isinstance(make_calibrator("isotonic"), IsotonicRegression)
    with pytest.raises(ValueError, match="unknown method"):
        make_calibrator("beta")


def test_a_temperature_on_the_edge_of_the_search_range_warns():
    confident = np.linspace(0.6, 0.9, 50)
    with pytest.warns(RuntimeWarning, match=r"edge of its range \[0\.01, 100\]"):
        low = TemperatureScaling().fit(confident, np.ones(50))  # always right: sharpen forever
    assert low.temperature == pytest.approx(0.01, rel=1e-4)
    assert at_temperature_bound(low.temperature)
    with pytest.warns(RuntimeWarning, match="not a fitted temperature"):
        high = TemperatureScaling().fit([0.9] * 40, [1, 0] * 20)  # right half the time: flatten
    assert high.temperature == pytest.approx(100.0, rel=1e-4)
    logits = np.tile([2.0, 0.0, -1.0], (40, 1))
    with pytest.warns(RuntimeWarning, match="edge of its range"):
        assert fit_temperature(logits, np.zeros(40, dtype=int)) == pytest.approx(0.01, rel=1e-4)


def test_an_ordinary_temperature_fit_is_silent():
    conf, correct = overconfident(9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = TemperatureScaling().fit(conf, correct)
    assert not at_temperature_bound(model.temperature)
    assert not at_temperature_bound(1.0) and at_temperature_bound(100.0)


def test_cross_fit_can_hand_back_the_map_of_each_fold():
    conf, correct = overconfident(10, n=500)
    models = []
    out = cross_fit("platt", conf, correct, n_folds=5, seed=2, models=models)
    folds = fold_ids(500, 5, seed=2)
    assert len(models) == 5
    for f, model in enumerate(models):
        assert out[folds == f] == pytest.approx(model.predict(conf[folds == f]), abs=1e-12)
