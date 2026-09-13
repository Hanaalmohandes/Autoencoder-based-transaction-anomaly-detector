"""Development-only training ablations; never scores or tunes on test labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transaction_anomaly.core import (evaluate, prepare, feature_matrix, split_data,
                                     train_autoencoder, reconstruction_scores, select_threshold)


def main():
    out = Path("reports/training-ablations.json")
    train, validation, _ = split_data(pd.read_csv("data/creditcard.csv"))
    boundary = train.Time.quantile(.8)
    fit, stop = train[train.Time < boundary], train[train.Time >= boundary]
    boundary = validation.Time.quantile(.5)
    cal = validation[validation.Time < boundary]
    selection = validation[validation.Time >= boundary]
    scaler, (x_fit, x_stop, x_cal) = prepare(fit, stop, cal)
    selected = json.loads(Path("outputs/high-recall-42/metrics.json").read_text())["models"]["autoencoder"]["selected"]
    ix = selected["feature_indices"]
    x_selection = scaler.transform(feature_matrix(selection)).astype(np.float32)[:, ix]
    configs = [
        {"name": "L2 regularization", "weight_decay": 1e-4},
        {"name": "Denoising dropout plus L2", "weight_decay": 1e-4, "input_dropout": .1},
        {"name": "25% normal subset; lower LR; smaller batch; longer patience",
         "fraction": .25, "learning_rate": 3e-4, "batch_size": 256, "patience": 10},
    ]
    rows = []
    for config in configs:
        config = config.copy()
        name, fraction = config.pop("name"), config.pop("fraction", 1)
        x = x_fit[:, ix]
        if fraction < 1:
            indices = np.random.default_rng(42).choice(len(x), int(len(x) * fraction), replace=False)
            x = x[indices]
        model, history = train_autoencoder(x, x_stop[stop.Class.to_numpy() == 0][:, ix],
            latent_dim=selected["latent_dim"], activation=selected["activation"], epochs=60, **config)
        a = reconstruction_scores(model, x_cal[:, ix], score_clip=selected["score_clip"])
        b = reconstruction_scores(model, x_selection, score_clip=selected["score_clip"])
        options = []
        for budget in [.00025, .0005, .00075, .001]:
            t = select_threshold(a[cal.Class.to_numpy() == 0], budget)
            options.append({"calibration_budget": budget, "selection": evaluate(selection.Class,b,t)})
        winner = max(options, key=lambda r: (r["selection"]["false_positive_rate"] <= .001,
                     r["selection"]["recall"], r["selection"]["precision"], r["selection"]["average_precision"]))
        rows.append({"name": name, "settings": config, "normal_training_rows": len(x),
                     "epochs_run": len(history), "threshold_options": options, "chosen": winner})
    out.write_text(json.dumps({"scope": "Development only; no final-test scoring or deployment",
                               "control": selected, "ablations": rows}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
