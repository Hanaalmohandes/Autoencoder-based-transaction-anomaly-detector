from argparse import Namespace
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import IsolationForest

from transaction_anomaly.cli import run, score
from transaction_anomaly.core import (FEATURES, evaluate, feature_matrix, prepare,
                                       select_threshold, split_data, validate_data,
                                       train_autoencoder, reconstruction_scores)


@pytest.fixture
def transactions():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(600, 28)), columns=FEATURES[:-1])
    frame["Amount"] = rng.lognormal(size=600)
    frame["Time"] = np.arange(600)
    frame["Class"] = (np.arange(600) % 20 == 0).astype(int)
    frame.loc[frame.Class == 1, "V1"] += 8
    return frame


def test_split_and_preprocessing_do_not_leak(transactions):
    train, val, test = split_data(pd.concat([transactions, transactions.iloc[:10]]))
    assert len(train) + len(val) + len(test) == len(transactions)
    assert train.Time.max() < val.Time.min() < test.Time.min()
    assert not set(train.index) & set(test.index)
    scaler, (x_train, _, _) = prepare(train, val, test)
    np.testing.assert_allclose(scaler.mean_, feature_matrix(train[train.Class == 0]).mean(axis=0))
    assert len(x_train) == (train.Class == 0).sum()
    val = val.copy()
    val["V1"] += 10000
    other_scaler, _ = prepare(train, val, test)
    np.testing.assert_array_equal(scaler.mean_, other_scaler.mean_)


@pytest.mark.parametrize("budget", [0, 0.001, 0.1, 0.5])
def test_threshold_respects_budget_with_ties(budget):
    scores = np.repeat(np.arange(10), 10)
    threshold = select_threshold(scores, budget)
    assert (scores > threshold).mean() <= budget


def test_known_metrics():
    result = evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.2, 0.9]), 0.5)
    assert result["recall"] == result["precision"] == result["false_positive_rate"] == 0.5
    assert result["false_positives"] == 1
    assert result["alerts"] == 2


def test_capped_residuals_limit_single_feature_dominance():
    class Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)
    x = np.array([[100, 0, 0], [2, 2, 2]], dtype=np.float32)
    original = reconstruction_scores(Zero(), x)
    robust = reconstruction_scores(Zero(), x, score_clip=4)
    assert original[0] > original[1]
    np.testing.assert_allclose(robust, [4 / 3, 4])


def test_validation_rejects_invalid_data(transactions):
    transactions.loc[0, "Amount"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        validate_data(transactions)


def test_quantile_transform_fits_only_normal_training(transactions):
    train, val, test = split_data(transactions)
    scaler, arrays = prepare(train, val, test, preprocessing="quantile")
    altered = train.copy()
    altered.loc[altered.Class == 1, "V1"] = 1e12
    other, _ = prepare(altered, val, test, preprocessing="quantile")
    np.testing.assert_array_equal(scaler.quantiles_, other.quantiles_)
    shifted = val.copy()
    shifted["V1"] = 1e12
    same, (_, x_shifted, _) = prepare(train, shifted, test, preprocessing="quantile")
    np.testing.assert_array_equal(scaler.quantiles_, same.quantiles_)
    assert np.isfinite(x_shifted).all()
    assert np.max(np.abs(x_shifted[:, 0])) < 6
    model, _ = train_autoencoder(arrays[0], arrays[1], epochs=1, latent_dim=2)
    assert reconstruction_scores(model, arrays[2]).shape == (len(test),)


def test_end_to_end_and_saved_inference(transactions, tmp_path):
    data = tmp_path / "data.csv"
    transactions.to_csv(data, index=False)
    output = tmp_path / "experiment"
    run(Namespace(data=data, output=output, epochs=2, max_fpr=0.05,
                  seed=42, split="chronological"))
    scored_path = tmp_path / "scored.csv"
    _, _, test = split_data(transactions)
    test[FEATURES].to_csv(tmp_path / "unlabeled.csv", index=False)
    score(Namespace(data=tmp_path / "unlabeled.csv", model_dir=output, output=scored_path))
    trained_scores = pd.read_csv(output / "test_scores.csv")
    loaded_scores = pd.read_csv(scored_path)
    for name in ("autoencoder", "isolation_forest"):
        np.testing.assert_allclose(loaded_scores[f"{name}_score"], trained_scores[f"{name}_score"], rtol=1e-6)
        np.testing.assert_array_equal(loaded_scores[f"{name}_alert"], trained_scores[f"{name}_alert"])


def test_robust_artifact_inference_with_different_scalers(transactions, tmp_path):
    train, val, test = split_data(transactions)
    scaler, (x_train, x_val, x_test) = prepare(train, val, test, "quantile")
    forest_scaler, (f_train, _, f_test) = prepare(train, val, test)
    model, _ = train_autoencoder(x_train, x_val, epochs=1, latent_dim=2)
    forest = IsolationForest(n_estimators=10, random_state=42).fit(f_train)
    torch.save(model.state_dict(), tmp_path / "autoencoder.pt")
    with (tmp_path / "preprocessing_and_baseline.pkl").open("wb") as f:
        pickle.dump({"scaler": scaler, "forest_scaler": forest_scaler, "forest": forest,
                     "latent_dim": 2, "score_clip": 1, "features": FEATURES,
                     "thresholds": {"autoencoder": 0.5, "isolation_forest": 0.5}}, f)
    test[FEATURES].to_csv(tmp_path / "input.csv", index=False)
    score(Namespace(data=tmp_path / "input.csv", model_dir=tmp_path, output=tmp_path / "scores.csv"))
    result = pd.read_csv(tmp_path / "scores.csv")
    np.testing.assert_allclose(result.autoencoder_score,
                               reconstruction_scores(model, x_test, score_clip=1), rtol=1e-6)
    np.testing.assert_allclose(result.isolation_forest_score, -forest.score_samples(f_test), rtol=1e-6)
