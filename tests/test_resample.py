import numpy as np
import pytest

from calibcheck.metrics import bin_stats
from calibcheck.resample import (
    bootstrap,
    cluster_indices,
    consistency_resampling,
    resample_indices,
)


def test_bootstrap_interval_of_a_mean_has_the_textbook_width():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.7).astype(float)
    lo, hi = bootstrap(lambda idx: {"mean": y[idx].mean()}, 400, n_resamples=2000, seed=1)["mean"]
    se = np.sqrt(y.mean() * (1 - y.mean()) / 400)
    assert lo < y.mean() < hi
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.1)


def test_bootstrap_is_reproducible_and_drops_undefined_draws():
    y = np.arange(50, dtype=float)

    def statistic(idx):
        return {"mean": y[idx].mean(), "never": float("nan")}

    a = bootstrap(statistic, 50, n_resamples=200, seed=3)
    b = bootstrap(statistic, 50, n_resamples=200, seed=3)
    c = bootstrap(statistic, 50, n_resamples=200, seed=4)
    assert a["mean"] == b["mean"] != c["mean"]
    assert np.isnan(a["never"]).all()


def test_cluster_bootstrap_keeps_groups_whole_and_widens_the_interval():
    rng = np.random.default_rng(5)
    per_group = rng.random(40)
    y = np.repeat(per_group, 10)  # ten copies of one value per group
    groups = np.repeat(np.arange(40), 10).tolist()

    def statistic(idx):
        return {"mean": y[idx].mean()}

    naive = bootstrap(statistic, 400, n_resamples=1500, seed=6)["mean"]
    clustered = bootstrap(statistic, 400, n_resamples=1500, seed=6, groups=groups)["mean"]
    ratio = (clustered[1] - clustered[0]) / (naive[1] - naive[0])
    assert ratio == pytest.approx(np.sqrt(10), rel=0.2)

    clusters = cluster_indices(["b", "a", "b", "c", "a"])
    assert [c.tolist() for c in clusters] == [[0, 2], [1, 4], [3]]
    draw = resample_indices(5, np.random.default_rng(7), clusters)
    members = {0: 0, 2: 0, 1: 1, 4: 1, 3: 2}
    for cluster in clusters:
        counts = [int(np.sum(draw == i)) for i in cluster]
        assert len(set(counts)) == 1, members


def test_bootstrap_rejects_bad_arguments():
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap(lambda idx: {}, 5, n_resamples=0)
    with pytest.raises(ValueError, match="level"):
        bootstrap(lambda idx: {}, 5, level=1.0)
    with pytest.raises(ValueError, match="one entry per item"):
        bootstrap(lambda idx: {}, 5, groups=["a"])


def calibrated(seed, n=400):
    rng = np.random.default_rng(seed)
    conf = np.round(rng.uniform(0.05, 0.95, size=n), 2)
    return conf, (rng.random(n) < conf).astype(float)


def test_consistency_bands_and_test_for_a_calibrated_model():
    conf, correct = calibrated(0, n=2000)
    result = consistency_resampling(conf, correct, n_bins=10, n_resamples=500, seed=1)
    stats = bin_stats(conf, correct, n_bins=10)
    m = stats.nonempty
    assert np.all(result.lower[m] <= result.upper[m])
    inside = (stats.accuracy[m] >= result.lower[m]) & (stats.accuracy[m] <= result.upper[m])
    assert inside.sum() >= m.sum() - 1
    assert np.all(
        (stats.confidence[m] >= result.lower[m]) & (stats.confidence[m] <= result.upper[m])
    )
    assert result.p_value > 0.05
    assert result.ece == pytest.approx(stats.ece())
    assert result.null_mean > 0  # the plug-in ECE is positive even under calibration


def test_consistency_test_rejects_a_miscalibrated_model():
    rng = np.random.default_rng(2)
    conf = np.round(rng.uniform(0.3, 0.95, size=1000), 2)
    correct = (rng.random(1000) < conf - 0.15).astype(float)
    result = consistency_resampling(conf, correct, n_bins=10, n_resamples=400, seed=3)
    assert result.p_value == pytest.approx(1 / 401)
    assert result.ece > result.null_quantile
    assert np.isnan(result.lower[0]) and np.isnan(result.upper[0])  # nothing below 0.1


def test_consistency_test_has_about_the_nominal_false_alarm_rate():
    rejections = 0
    for seed in range(200):
        conf, correct = calibrated(100 + seed, n=300)
        result = consistency_resampling(conf, correct, n_bins=10, n_resamples=200, seed=seed)
        rejections += result.p_value <= 0.05
    assert 2 <= rejections <= 20  # 5 % of 200 is 10


def test_consistency_resampling_with_equal_mass_bins_and_bad_arguments():
    conf, correct = calibrated(9)
    result = consistency_resampling(conf, correct, strategy="mass", n_resamples=50, seed=0)
    assert not np.isnan(result.lower).any()
    assert result.n_resamples == 50 and result.level == 0.95
    with pytest.raises(ValueError, match="n_resamples"):
        consistency_resampling(conf, correct, n_resamples=0)
    with pytest.raises(ValueError, match="level"):
        consistency_resampling(conf, correct, level=0.0)
