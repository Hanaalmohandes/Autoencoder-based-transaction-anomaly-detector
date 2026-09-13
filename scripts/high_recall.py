"""Label-informed input selection with normal-only model fitting.

Choose on development data; evaluate the reused test only after development
reaches 70% recall within a 0.1% false-positive budget.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from transaction_anomaly.core import (FEATURES, evaluate, feature_matrix, prepare,
    reconstruction_scores, select_threshold, split_data, train_autoencoder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/high-recall-42"))
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise ValueError("Choose a new output directory")
    out.mkdir(parents=True)
    train, val, test = split_data(pd.read_csv("data/creditcard.csv"))
    boundary = train.Time.quantile(.8)
    fit, stop = train[train.Time < boundary], train[train.Time >= boundary]
    boundary = val.Time.quantile(.5)
    cal, selection = val[val.Time < boundary], val[val.Time >= boundary]
    scaler, (x_fit, x_stop, x_cal) = prepare(fit, stop, cal)
    x_selection = scaler.transform(feature_matrix(selection)).astype(np.float32)
    # Feature relevance uses fitting-period labels only, never validation/test.
    # This is explicitly label-informed selection, not purely unsupervised learning.
    fit_all = scaler.transform(feature_matrix(fit))
    rank_scores = [average_precision_score(fit.Class, np.abs(fit_all[:, j]))
                   for j in range(len(FEATURES))]
    ranking = np.argsort(-np.array(rank_scores)).tolist()
    protocol = {
        "minimum_recall": .7, "max_fpr": .001, "seed": 42, "epochs": 60,
        "model_configs": [[4, 2], [8, 2], [8, 4], [29, 8]],
        "activations": ["relu", "tanh"], "score_clips": [None, 4, 9],
        "calibration_budgets": [.00025, .0005, .00075, .001],
        "forest_max_samples": [256, 1024],
        "feature_selection": "fitting-label average precision of absolute normal-standardized features",
        "feature_ranking": [{"feature": FEATURES[j], "fit_ap": rank_scores[j]} for j in ranking],
        "test_status": "reused historical test; exploratory, not an independent holdout",
        "selection_rule": "FPR eligibility, then recall, then precision, then average precision",
        "dataset_sha256": hashlib.sha256(Path("data/creditcard.csv").read_bytes()).hexdigest(),
        "partitions": {name: {"rows": len(p), "fraud": int(p.Class.sum()),
            "time_min": float(p.Time.min()), "time_max": float(p.Time.max())}
            for name, p in [("fit", fit), ("early_stop", stop), ("calibration", cal),
                            ("selection", selection), ("test", test)]},
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    candidates, best = [], {}

    def consider(kind, model, indices, config, history=None):
        clips = protocol["score_clips"] if kind == "autoencoder" else [None]
        for clip in clips:
            def scores(x):
                return (reconstruction_scores(model, x[:, indices], score_clip=clip)
                        if kind == "autoencoder" else -model.score_samples(x[:, indices]))
            cal_scores, sel_scores = scores(x_cal), scores(x_selection)
            for budget in protocol["calibration_budgets"]:
                threshold = select_threshold(cal_scores[cal.Class.to_numpy() == 0], budget)
                metrics = evaluate(selection.Class, sel_scores, threshold)
                record = {"model": kind, **config, "feature_indices": indices,
                          "input_features": [FEATURES[j] for j in indices],
                          "score_clip": clip, "calibration_budget": budget,
                          "calibration": evaluate(cal.Class, cal_scores, threshold),
                          "selection": metrics}
                candidates.append(record)
                key = (metrics["false_positive_rate"] <= .001, metrics["recall"],
                       metrics["precision"], metrics["average_precision"])
                if kind not in best or key > best[kind][0]:
                    best[kind] = (key, model, record, history)
                    print("BEST", json.dumps(record), flush=True)

    for n_features, latent in protocol["model_configs"]:
        indices = ranking[:n_features] if n_features < len(FEATURES) else list(range(len(FEATURES)))
        for activation in protocol["activations"]:
            model, history = train_autoencoder(x_fit[:, indices],
                x_stop[stop.Class.to_numpy() == 0][:, indices], epochs=60,
                latent_dim=latent, activation=activation)
            consider("autoencoder", model, indices,
                     {"latent_dim": latent, "activation": activation}, history)
    for n_features in [4, 8, 29]:
        indices = ranking[:n_features] if n_features < len(FEATURES) else list(range(len(FEATURES)))
        for size in protocol["forest_max_samples"]:
            model = IsolationForest(n_estimators=300, max_samples=size,
                                     random_state=42, n_jobs=2).fit(x_fit[:, indices])
            consider("isolation_forest", model, indices, {"max_samples": size})
    (out / "development.json").write_text(json.dumps(candidates, indent=2))
    report = {"protocol": protocol, "models": {k: {"selected": v[2]} for k, v in best.items()}}
    ae = best["autoencoder"]
    passes = ae[2]["selection"]["recall"] >= .7 and ae[2]["selection"]["false_positive_rate"] <= .001
    report["development_target_met"] = passes
    # Save the frozen candidate even if the development target was not met.
    torch.save(ae[1].state_dict(), out / "autoencoder.pt")
    forest = best["isolation_forest"]
    with (out / "preprocessing_and_baseline.pkl").open("wb") as f:
        pickle.dump({"scaler": scaler, "forest": forest[1], "features": FEATURES,
            "feature_indices": ae[2]["feature_indices"], "forest_feature_indices": forest[2]["feature_indices"],
            "latent_dim": ae[2]["latent_dim"], "activation": ae[2]["activation"],
            "score_clip": ae[2]["score_clip"],
            "thresholds": {k: v[2]["selection"]["threshold"] for k, v in best.items()}}, f)
    pd.DataFrame(ae[3]).to_csv(out / "training_history.csv", index=False)
    if passes:
        x_test = scaler.transform(feature_matrix(test)).astype(np.float32)
        output_scores = test[["Time", "Class"]].copy()
        for kind, (_, model, record, _) in best.items():
            x = x_test[:, record["feature_indices"]]
            values = (reconstruction_scores(model, x, score_clip=record["score_clip"])
                      if kind == "autoencoder" else -model.score_samples(x))
            threshold = record["selection"]["threshold"]
            report["models"][kind]["test"] = evaluate(test.Class, values, threshold)
            output_scores[kind + "_score"] = values
            output_scores[kind + "_alert"] = values > threshold
        output_scores.to_csv(out / "test_scores.csv", index=False)
        m = report["models"]["autoencoder"]["test"]
        b = report["models"]["isolation_forest"]["test"]
        report["recall_target_observed_on_reused_test"] = m["recall"] >= .7 and m["false_positive_rate"] <= .001
        report["beats_tuned_baseline_recall"] = m["recall"] > b["recall"]
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
