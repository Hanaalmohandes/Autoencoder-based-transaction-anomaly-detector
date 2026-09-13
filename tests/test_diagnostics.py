import numpy as np
import pytest

from transaction_anomaly.core import evaluate
from transaction_anomaly.diagnostics import ResidualScorers, optimal_threshold, threshold_sweep


@pytest.mark.parametrize("budget", [0, .1, .5])
def test_threshold_sweep_matches_brute_force_with_ties(budget):
    y = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    s = np.array([1, 1, 2, 3, 4, 5, 5, 6], dtype=float)
    curve = threshold_sweep(y, s)
    for record in curve.to_dict("records"):
        expected = evaluate(y, s, record["threshold"])
        for metric in ["recall", "precision", "f1", "false_positives", "false_positive_rate"]:
            assert record[metric] == pytest.approx(expected[metric])
    best = optimal_threshold(y, s, budget)
    assert best["false_positive_rate"] <= budget
    assert best["recall"] == curve[curve.false_positive_rate <= budget].recall.max()


def test_score_normalizers_are_frozen_and_finite():
    rng = np.random.default_rng(42)
    r, z = rng.normal(size=(100, 5)), rng.normal(size=(100, 2))
    r[:, 0] = 0  # Constant residual feature must not cause division by zero.
    scorer = ResidualScorers().fit(r, z)
    mean = scorer.mean.copy()
    a = scorer.score(r[:3], z[:3])
    scorer.score(r[3:6] * 1000, z[3:6] * 1000)
    b = scorer.score(r[:3], z[:3])
    np.testing.assert_array_equal(mean, scorer.mean)
    assert len(a) == 12
    for key in a:
        assert np.isfinite(a[key]).all()
        np.testing.assert_array_equal(a[key], b[key])


def test_invalid_threshold_inputs():
    with pytest.raises(ValueError):
        optimal_threshold([0, 1], [0, np.nan])
    with pytest.raises(ValueError):
        optimal_threshold([0, 1], [0, 1], 1)
