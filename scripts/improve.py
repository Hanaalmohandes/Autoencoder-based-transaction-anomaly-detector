"""Prespecified development search; evaluate selected models on reused test once."""
import argparse
import hashlib
import json
import platform
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.ensemble import IsolationForest

from transaction_anomaly.core import (FEATURES, evaluate, prepare, reconstruction_scores,
                                       select_threshold, split_data, train_autoencoder,
                                       feature_matrix)

OUT = Path("outputs/calibrated-42")
BUDGET = 0.001


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output
    execute(output)


def execute(OUT):
    if OUT.exists():
        raise ValueError("Choose a new output directory to preserve previous experiments")
    OUT.mkdir(parents=True)
    train, validation, test = split_data(pd.read_csv("data/creditcard.csv"))
    # Early stopping uses a tail of the original training partition. Threshold
    # calibration and model selection use separate halves of original validation.
    cut = train.Time.quantile(0.8)
    fit, stop = train[train.Time < cut], train[train.Time >= cut]
    cut = validation.Time.quantile(0.5)
    cal, select = validation[validation.Time < cut], validation[validation.Time >= cut]
    protocol = {"seed": 42, "max_fpr": BUDGET, "epochs": 30,
                "dataset_sha256": hashlib.sha256(Path("data/creditcard.csv").read_bytes()).hexdigest(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "torch": torch.__version__, "sklearn": sklearn.__version__},
                "autoencoder_latent_dims": [2, 4, 8],
                "forest_max_samples": [256, 1024],
                "preprocessing": ["standard", "quantile"],
                "score_clips": [None, 1, 4, 9],
                "calibration_budgets": [0.0005, 0.00075, 0.001],
                "selection": "highest selection recall within FPR cap; then average precision",
                "test_status": "Previously inspected test; exploratory confirmation, not fresh holdout",
                "partitions": {name: {"rows": len(p), "fraud": int(p.Class.sum()),
                                        "time_min": float(p.Time.min()), "time_max": float(p.Time.max())}
                               for name, p in [("fit", fit), ("early_stop", stop),
                                               ("calibration", cal), ("selection", select), ("test", test)]}}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    candidates = []
    best = {}
    for preprocessing in protocol["preprocessing"]:
        scaler, (x_fit, x_stop, x_cal) = prepare(fit, stop, cal, preprocessing)
        x_select = scaler.transform(feature_matrix(select)).astype(np.float32)
        for kind in ["autoencoder", "isolation_forest"]:
            for setting in (protocol["autoencoder_latent_dims"] if kind == "autoencoder"
                            else protocol["forest_max_samples"]):
                if kind == "autoencoder":
                    model, history = train_autoencoder(x_fit, x_stop[stop.Class.to_numpy() == 0],
                                                       latent_dim=setting, epochs=30)
                else:
                    model = IsolationForest(n_estimators=300, max_samples=setting,
                                            random_state=42, n_jobs=2).fit(x_fit)
                    history = None
                for clip in protocol["score_clips"] if kind == "autoencoder" else [None]:
                    def get_scores(x):
                        return (reconstruction_scores(model, x, score_clip=clip)
                                if kind == "autoencoder" else -model.score_samples(x))
                    cal_scores = get_scores(x_cal)
                    selection_scores = get_scores(x_select)
                    for calibration_budget in protocol["calibration_budgets"]:
                        threshold = select_threshold(cal_scores[cal.Class.to_numpy() == 0], calibration_budget)
                        metrics = evaluate(select.Class, selection_scores, threshold)
                        record = {"model": kind, "preprocessing": preprocessing, "setting": setting,
                                  "score_clip": clip, "calibration_budget": calibration_budget,
                                  "calibration": evaluate(cal.Class, cal_scores, threshold), "selection": metrics}
                        candidates.append(record)
                        print(json.dumps(record), flush=True)
                        key = (metrics["false_positive_rate"] <= BUDGET, metrics["recall"],
                               metrics["average_precision"])
                        if kind not in best or key > best[kind][0]:
                            best[kind] = (key, model, scaler, threshold, record, history)
    # Persist development results before inspecting any test predictions.
    (OUT / "development.json").write_text(json.dumps(candidates, indent=2))
    report = {"protocol": protocol, "models": {}}
    scores = test[["Time", "Class"]].copy()
    for kind, (_, model, scaler, threshold, record, history) in best.items():
        x_test = scaler.transform(feature_matrix(test)).astype(np.float32)
        values = (reconstruction_scores(model, x_test, score_clip=record["score_clip"])
                  if kind == "autoencoder" else -model.score_samples(x_test))
        report["models"][kind] = {"selected": record, "test": evaluate(test.Class, values, threshold)}
        scores[kind + "_score"] = values
        scores[kind + "_alert"] = values > threshold
    ae = report["models"]["autoencoder"]["test"]
    forest = report["models"]["isolation_forest"]["test"]
    report["target_observed_on_reused_test"] = ae["recall"] > forest["recall"] and ae["false_positive_rate"] <= BUDGET
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    scores.to_csv(OUT / "test_scores.csv", index=False)
    ae = best["autoencoder"]
    forest = best["isolation_forest"]
    torch.save(ae[1].state_dict(), OUT / "autoencoder.pt")
    with (OUT / "preprocessing_and_baseline.pkl").open("wb") as f:
        pickle.dump({"scaler": ae[2], "forest_scaler": forest[2], "forest": forest[1],
                     "latent_dim": ae[4]["setting"], "features": FEATURES,
                     "score_clip": ae[4]["score_clip"],
                     "thresholds": {"autoencoder": ae[3], "isolation_forest": forest[3]}}, f)
    pd.DataFrame(ae[5]).to_csv(OUT / "training_history.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
