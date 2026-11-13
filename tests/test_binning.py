import numpy as np
import pytest

from calibcheck.binning import assign_bins, equal_mass_edges, equal_width_edges, make_edges


def test_equal_width_edges_are_the_decimal_fractions():
    edges = equal_width_edges(10)
    assert edges.tolist() == [i / 10 for i in range(11)]
    assert edges[3] == 0.3  # linspace would give 0.30000000000000004


@pytest.mark.parametrize("n_bins", [5, 10, 15, 20, 100])
def test_value_on_an_edge_goes_to_the_bin_that_starts_there(n_bins):
    edges = equal_width_edges(n_bins)
    on_edges = np.array([i / n_bins for i in range(n_bins)])
    assert assign_bins(on_edges, edges).tolist() == list(range(n_bins))


def test_decimal_literals_that_break_floor_multiplication():
    # int(0.57 * 100) == 56, yet 0.57 is not below the edge 57/100.
    assert int(0.57 * 100) == 56
    assert assign_bins([0.57], equal_width_edges(100)).tolist() == [57]
    assert assign_bins([0.29, 0.3, 0.7], equal_width_edges(10)).tolist() == [2, 3, 7]


def test_zero_and_one_fall_in_the_end_bins():
    edges = equal_width_edges(15)
    assert assign_bins([0.0, 1.0], edges).tolist() == [0, 14]


def test_single_bin_holds_everything():
    edges = equal_width_edges(1)
    assert edges.tolist() == [0.0, 1.0]
    assert assign_bins([0.0, 0.4, 1.0], edges).tolist() == [0, 0, 0]


def test_equal_mass_without_ties_gives_counts_within_one():
    rng = np.random.default_rng(0)
    values = rng.random(1003)
    edges = equal_mass_edges(values, 10)
    counts = np.bincount(assign_bins(values, edges), minlength=10)
    assert len(edges) == 11
    assert counts.sum() == 1003
    assert counts.max() - counts.min() <= 1


def test_equal_mass_keeps_ties_in_one_bin():
    values = np.array([0.5] * 10 + [0.8] * 30 + [0.9] * 40 + [1.0] * 20)
    edges = equal_mass_edges(values, 10)
    assert edges.tolist() == [0.0, 0.8, 0.9, 1.0, 1.0]
    idx = assign_bins(values, edges)
    for level in (0.5, 0.8, 0.9, 1.0):
        assert len(set(idx[values == level].tolist())) == 1
    assert np.bincount(idx).tolist() == [10, 30, 40, 20]


def test_equal_mass_never_returns_an_empty_bin():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 60))
        levels = rng.choice([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0], size=n)
        n_bins = int(rng.integers(1, 25))
        edges = equal_mass_edges(levels, n_bins)
        counts = np.bincount(assign_bins(levels, edges), minlength=len(edges) - 1)
        assert len(edges) - 1 <= n_bins
        assert counts.min() >= 1
        assert counts.sum() == n


def test_equal_mass_does_not_depend_on_input_order():
    rng = np.random.default_rng(2)
    values = np.round(rng.random(500), 2)
    shuffled = rng.permutation(values)
    assert equal_mass_edges(values, 12).tolist() == equal_mass_edges(shuffled, 12).tolist()


def test_equal_mass_with_identical_values_is_one_bin():
    edges = equal_mass_edges([0.7] * 50, 10)
    assert edges.tolist() == [0.0, 1.0]


def test_equal_mass_with_fewer_items_than_bins():
    values = [0.2, 0.9, 0.6]
    edges = equal_mass_edges(values, 10)
    assert edges.tolist() == [0.0, 0.6, 0.9, 1.0]
    assert assign_bins(values, edges).tolist() == [0, 2, 1]


def test_equal_mass_rejects_bad_input():
    with pytest.raises(ValueError, match="zero values"):
        equal_mass_edges([], 5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        equal_mass_edges([0.2, 1.2], 5)
    with pytest.raises(ValueError, match="NaN"):
        equal_mass_edges([0.2, float("nan")], 5)


@pytest.mark.parametrize("bad", [0, -3])
def test_n_bins_must_be_positive(bad):
    with pytest.raises(ValueError, match="at least 1"):
        equal_width_edges(bad)


@pytest.mark.parametrize("bad", [2.5, "10", True])
def test_n_bins_must_be_an_integer(bad):
    with pytest.raises(TypeError):
        equal_width_edges(bad)


def test_make_edges_dispatches_and_rejects_unknown_strategy():
    assert make_edges([0.1, 0.9], 4, "width").tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert make_edges([0.1, 0.9], 2, "mass").tolist() == [0.0, 0.9, 1.0]
    with pytest.raises(ValueError, match="unknown binning strategy"):
        make_edges([0.1], 4, "quantile")


def test_assign_bins_rejects_malformed_edges():
    with pytest.raises(ValueError):
        assign_bins([0.5], [0.0])
    with pytest.raises(ValueError):
        assign_bins([0.5], [0.0, 0.6, 0.4, 1.0])
