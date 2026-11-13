import math

import numpy as np
import pytest
from sklearn import metrics as sk

from calibcheck.metrics import brier_score, ece
from calibcheck.multiclass import (
    check_probabilities,
    classwise_ece,
    multiclass_brier,
    multiclass_log_loss,
    pad_vectors,
    softmax,
    top_label,
    vector_summary,
)


def random_problem(seed, n=400, k=4):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(k), size=n)
    labels = np.array([rng.choice(k, p=p) for p in probs])
    return probs, labels


def test_softmax_rows_sum_to_one_and_ignore_shifts_and_padding():
    z = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0], [0.0, -math.inf, 0.0]])
    p = softmax(z)
    assert p.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])
    assert p[0] == pytest.approx(p[1])
    assert p[2].tolist() == [0.5, 0.0, 0.5]
    assert softmax([0.0, math.log(3.0)])[0] == pytest.approx([0.25, 0.75])


def test_top_label_confidence_and_correctness():
    probs = [[0.7, 0.2, 0.1], [0.1, 0.5, 0.4], [0.4, 0.4, 0.2]]
    conf, correct = top_label(probs, [0, 2, 0])
    assert conf.tolist() == [0.7, 0.5, 0.4]
    assert correct.tolist() == [1.0, 0.0, 1.0]  # the tie goes to class 0


def test_two_class_classwise_ece_is_the_binary_ece():
    rng = np.random.default_rng(0)
    p1 = rng.random(500)
    labels = (rng.random(500) < p1**2).astype(int)
    probs = np.column_stack([1 - p1, p1])
    binary = ece(p1, labels, n_bins=10)
    assert classwise_ece(probs, labels, n_bins=10) == pytest.approx(binary, abs=1e-12)


def test_classwise_ece_is_the_mean_of_per_class_errors():
    probs, labels = random_problem(1)
    per_class = [ece(probs[:, k], labels == k, n_bins=12, strategy="mass") for k in range(4)]
    got = classwise_ece(probs, labels, n_bins=12, strategy="mass")
    assert got == pytest.approx(np.mean(per_class), abs=1e-12)
    assert classwise_ece(probs, labels, debias=True) < classwise_ece(probs, labels)


def test_brier_and_log_loss_of_vectors():
    probs, labels = random_problem(2)
    assert multiclass_log_loss(probs, labels) == pytest.approx(
        sk.log_loss(labels, probs, labels=[0, 1, 2, 3]), abs=1e-12
    )
    onehot = np.eye(4)[labels]
    assert multiclass_brier(probs, labels) == pytest.approx(np.mean(((probs - onehot) ** 2).sum(1)))
    p1 = probs[:, 0] / (probs[:, 0] + probs[:, 1])
    two = np.column_stack([1 - p1, p1])
    y = (labels % 2 == 1).astype(int)
    assert multiclass_brier(two, y) == pytest.approx(2 * brier_score(p1, y))
    assert multiclass_log_loss([[1.0, 0.0]], [1]) == pytest.approx(-math.log(1e-12))


def test_vector_summary_skips_classwise_terms_for_ragged_input():
    probs, labels = random_problem(3)
    full = vector_summary(probs, labels)
    assert set(full) == {"nll", "brier_multiclass", "classwise_ece", "classwise_ece_debiased"}
    assert set(vector_summary(probs, labels, classwise=False)) == {"nll", "brier_multiclass"}


def test_pad_vectors():
    out, ragged = pad_vectors([[0.5, 0.5], [0.2, 0.3, 0.5]], 0.0)
    assert ragged and out.tolist() == [[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]
    out, ragged = pad_vectors([[1.0, 2.0], [3.0, 4.0]], -math.inf)
    assert not ragged and out.shape == (2, 2)
    with pytest.raises(ValueError, match="no items"):
        pad_vectors([], 0.0)
    with pytest.raises(ValueError, match="at least two"):
        pad_vectors([[1.0]], 0.0)


@pytest.mark.parametrize(
    ("probs", "labels", "message"),
    [
        ([[0.5, 0.6]], [0], "sum to 1"),
        ([[1.2, -0.2]], [0], "non-negative"),
        ([[0.5, 0.5]], [2], r"\[0, K\)"),
        ([[0.5, 0.5]], [0.5], "integer"),
        ([[0.5, 0.5]], [0, 1], "one class index per item"),
        ([0.5, 0.5], [0], "matrix"),
    ],
)
def test_invalid_vectors_are_rejected(probs, labels, message):
    with pytest.raises(ValueError, match=message):
        check_probabilities(probs, labels)
