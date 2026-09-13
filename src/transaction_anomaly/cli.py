"""Download the public benchmark, run an experiment, or score new transactions."""

import argparse
import hashlib
import json
import pickle
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.datasets import fetch_openml
from sklearn.ensemble import IsolationForest

from .core import (FEATURES, Autoencoder, evaluate, feature_matrix, prepare,
                   reconstruction_scores, select_threshold, split_data,
                   train_autoencoder, validate_data)


def download(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # OpenML marks Time as a row identifier and otherwise drops it. Request it
    # as the retrieval target to retain it in frame; Class remains a column.
    # This changes retrieval only, never the modeling label.
    dataset = fetch_openml(data_id=1597, as_frame=True, target_column="Time",
                           data_home=str(args.output.parent / "openml_cache"))
    frame = dataset.frame
    frame["Class"] = frame["Class"].astype(int)
    validate_data(frame)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame):,} transactions to {args.output}")


def run(args):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if not 0 <= args.max_fpr < 1:
        raise ValueError("max-fpr must be in [0, 1)")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory")
    frame = pd.read_csv(args.data)
    train, val, test = split_data(frame, args.seed, args.split)
    scaler, (x_train, x_val, x_test) = prepare(train, val, test)
    y_val, y_test = val.Class.to_numpy(int), test.Class.to_numpy(int)
    model, history = train_autoencoder(x_train, x_val[y_val == 0],
                                       epochs=args.epochs, seed=args.seed)
    forest = IsolationForest(n_estimators=300, max_samples=256,
                             random_state=args.seed, n_jobs=2).fit(x_train)
    score_pairs = {
        "autoencoder": (reconstruction_scores(model, x_val), reconstruction_scores(model, x_test)),
        "isolation_forest": (-forest.score_samples(x_val), -forest.score_samples(x_test)),
    }
    report = {
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "config": {"seed": args.seed, "split": args.split, "max_fpr": args.max_fpr,
                   "epochs": args.epochs, "features": FEATURES},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "duplicates_removed": len(frame) - len(train) - len(val) - len(test),
        "splits": {name: {"rows": len(part), "fraud": int(part.Class.sum()),
                           "time_min": float(part.Time.min()), "time_max": float(part.Time.max())}
                   for name, part in zip(("train", "validation", "test"), (train, val, test))},
        "normal_training_rows": len(x_train), "models": {},
    }
    scored = test[["Time", "Class"]].copy()
    scored.insert(0, "source_row", test.index)
    thresholds = {}
    for name, (val_scores, test_scores) in score_pairs.items():
        threshold = select_threshold(val_scores[y_val == 0], args.max_fpr)
        thresholds[name] = threshold
        report["models"][name] = {"validation": evaluate(y_val, val_scores, threshold),
                                   "test": evaluate(y_test, test_scores, threshold)}
        scored[f"{name}_score"] = test_scores
        scored[f"{name}_alert"] = test_scores > threshold
    ae = report["models"]["autoencoder"]["test"]
    baseline = report["models"]["isolation_forest"]["test"]
    report["comparison"] = {
        "recall_difference": ae["recall"] - baseline["recall"],
        "autoencoder_within_test_fpr_budget": ae["false_positive_rate"] <= args.max_fpr,
        "baseline_within_test_fpr_budget": baseline["false_positive_rate"] <= args.max_fpr,
        "target_observed_on_this_split": ae["recall"] > baseline["recall"] and
                                         ae["false_positive_rate"] <= args.max_fpr,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    scored.to_csv(args.output / "test_scores.csv", index=False)
    torch.save(model.state_dict(), args.output / "autoencoder.pt")
    with (args.output / "preprocessing_and_baseline.pkl").open("wb") as handle:
        pickle.dump({"scaler": scaler, "forest": forest, "thresholds": thresholds,
                     "features": FEATURES}, handle)
    print(f"\nSaved experiment to {args.output}")
    for name, result in report["models"].items():
        m = result["test"]
        print(f"{name}: recall={m['recall']:.3%}, precision={m['precision']:.3%}, "
              f"FPR={m['false_positive_rate']:.3%}, alerts={m['alerts']}")
    print("Target observed on this split:", report["comparison"]["target_observed_on_this_split"])


def predict_frame(frame, model_dir):
    """Apply saved preprocessing, model, scoring rule, and frozen thresholds."""
    validate_data(frame, labeled=False)
    # Only load locally generated, trusted artifacts: pickle can execute code.
    with (model_dir / "preprocessing_and_baseline.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    if bundle["features"] != FEATURES:
        raise ValueError("Model feature schema does not match this version")
    x = bundle["scaler"].transform(feature_matrix(frame)).astype(np.float32)
    x = x[:, bundle.get("feature_indices", list(range(len(FEATURES))))]
    model = Autoencoder(input_dim=x.shape[1], latent_dim=bundle.get("latent_dim", 8),
                        activation=bundle.get("activation", "relu"))
    model.load_state_dict(torch.load(model_dir / "autoencoder.pt", weights_only=True,
                                     map_location="cpu"))
    result = frame.copy()
    forest_x = bundle.get("forest_scaler", bundle["scaler"]).transform(feature_matrix(frame)).astype(np.float32)
    forest_x = forest_x[:, bundle.get("forest_feature_indices", list(range(len(FEATURES))))]
    for name, scores in {"autoencoder": reconstruction_scores(model, x, score_clip=bundle.get("score_clip")),
                         "isolation_forest": -bundle["forest"].score_samples(forest_x)}.items():
        result[f"{name}_score"] = scores
        result[f"{name}_alert"] = scores > bundle["thresholds"][name]
    return result, bundle["thresholds"]


def score(args):
    frame = pd.read_csv(args.data)
    result, _ = predict_frame(frame, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved scores for {len(frame):,} transactions to {args.output}")


def evaluate_holdout(args):
    frame = pd.read_csv(args.data)
    validate_data(frame)
    if set(frame.Class.unique()) != {0, 1}:
        raise ValueError("Holdout evaluation requires both classes")
    result, thresholds = predict_frame(frame, args.model_dir)
    metrics = {name: evaluate(frame.Class, result[name + "_score"], thresholds[name])
               for name in ("autoencoder", "isolation_forest")}
    report = {"data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "model_directory": str(args.model_dir), "rows": len(frame),
              "thresholds_refitted": False, "models": metrics,
              "note": "Caller must ensure this labeled dataset is a genuinely fresh holdout."}
    if args.output.exists():
        raise ValueError("Choose a new output path to preserve prior evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help="Fetch OpenML creditcard dataset 1597")
    fetch.add_argument("--output", type=Path, default=Path("data/creditcard.csv"))
    fetch.set_defaults(func=download)
    train = sub.add_parser("train", help="Train and evaluate both normal-only detectors")
    train.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    train.add_argument("--output", type=Path, default=Path("outputs/run-42"))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--max-fpr", type=float, default=0.001)
    train.add_argument("--split", choices=["chronological", "stratified"], default="chronological")
    train.set_defaults(func=run)
    predict = sub.add_parser("score", help="Score a CSV with trusted saved model artifacts")
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=Path("outputs/scored.csv"))
    predict.set_defaults(func=score)
    holdout = sub.add_parser("evaluate", help="Evaluate a labeled holdout with frozen models and thresholds")
    holdout.add_argument("--data", type=Path, required=True)
    holdout.add_argument("--model-dir", type=Path, required=True)
    holdout.add_argument("--output", type=Path, required=True)
    holdout.set_defaults(func=evaluate_holdout)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
