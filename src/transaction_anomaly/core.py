"""Leakage-aware data preparation, normal-only training, and evaluation."""

import copy
import math
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def validate_data(frame, labeled=True):
    required = FEATURES + (["Class", "Time"] if labeled else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    values = frame[required].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Required columns must contain finite numeric values")
    if (frame.Amount < 0).any():
        raise ValueError("Amount must be nonnegative")
    if labeled and not frame.Class.isin([0, 1]).all():
        raise ValueError("Class must contain only 0 (normal) and 1 (fraud)")


def split_data(frame, seed=42, strategy="chronological"):
    validate_data(frame)
    # Remove exact duplicates before splitting, so repeated rows cannot leak.
    frame = frame.drop_duplicates(subset=["Time", *FEATURES, "Class"])
    if strategy == "chronological":
        frame = frame.sort_values("Time", kind="stable")
        # Keep equal timestamps together at boundaries.
        a = frame.Time.iloc[int(len(frame) * 0.6)]
        b = frame.Time.iloc[int(len(frame) * 0.8)]
        splits = (frame[frame.Time < a], frame[(frame.Time >= a) & (frame.Time < b)],
                  frame[frame.Time >= b])
    elif strategy == "stratified":
        train, holdout = train_test_split(frame, test_size=0.4, random_state=seed,
                                         stratify=frame.Class)
        val, test = train_test_split(holdout, test_size=0.5, random_state=seed,
                                    stratify=holdout.Class)
        splits = train, val, test
    else:
        raise ValueError("Unknown split strategy")
    for name, part in zip(("train", "validation", "test"), splits):
        if set(part.Class.unique()) != {0, 1}:
            raise ValueError(f"{name} split must contain both classes; use more data or another split")
    return splits


def feature_matrix(frame):
    x = frame[FEATURES].to_numpy(dtype=np.float64, copy=True)
    x[:, -1] = np.log1p(x[:, -1])
    return x


def prepare(train, val, test, preprocessing="standard"):
    normal = train[train.Class == 0]
    if preprocessing == "standard":
        scaler = StandardScaler()
    elif preprocessing == "quantile":
        scaler = QuantileTransformer(n_quantiles=min(1000, len(normal)),
                                     output_distribution="normal", random_state=42,
                                     subsample=None)
    else:
        raise ValueError("Unknown preprocessing method")
    scaler.fit(feature_matrix(normal))
    arrays = [scaler.transform(feature_matrix(p)).astype(np.float32)
              for p in (normal, val, test)]
    return scaler, arrays


class Autoencoder(nn.Module):
    def __init__(self, input_dim=len(FEATURES), latent_dim=8, activation="relu"):
        super().__init__()
        if activation not in {"relu", "tanh"}:
            raise ValueError("Unknown activation")
        nonlinearity = nn.ReLU if activation == "relu" else nn.Tanh
        self.network = nn.Sequential(
            nn.Linear(input_dim, 24), nonlinearity(), nn.Linear(24, latent_dim), nonlinearity(),
            nn.Linear(latent_dim, 24), nonlinearity(), nn.Linear(24, input_dim),
        )

    def forward(self, x):
        return self.network(x)


def reconstruction_scores(model, x, batch_size=4096, score_clip=None):
    """Mean squared residual, optionally capped per feature before averaging."""
    if score_clip is not None and (not math.isfinite(score_clip) or score_clip <= 0):
        raise ValueError("score_clip must be finite and positive")
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            ((model(batch) - batch) ** 2).clamp(max=score_clip).mean(dim=1).numpy()
            if score_clip is not None else ((model(batch) - batch) ** 2).mean(dim=1).numpy()
            for batch in torch.from_numpy(x).split(batch_size)
        ])


def train_autoencoder(x_train, x_val_normal, epochs=30, batch_size=512,
                      learning_rate=1e-3, patience=5, seed=42, latent_dim=8, activation="relu",
                      weight_decay=0.0, input_dropout=0.0):
    if not 0 <= input_dropout < 1 or weight_decay < 0:
        raise ValueError("Invalid training regularization")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    model = Autoencoder(x_train.shape[1], latent_dim=latent_dim, activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train)), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_loss, stale, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            corrupted = nn.functional.dropout(batch, p=input_dropout, training=True)
            loss = ((model(corrupted) - batch) ** 2).mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        val_loss = float(reconstruction_scores(model, x_val_normal).mean())
        if not math.isfinite(val_loss):
            raise ValueError("Training produced non-finite reconstruction error")
        history.append({"epoch": epoch, "train_mse": total / len(x_train), "val_mse": val_loss})
        print(f"Epoch {epoch:02d}: train={total / len(x_train):.5f}, val={val_loss:.5f}", flush=True)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def select_threshold(normal_scores, max_fpr):
    """Largest allowable alert set under an empirical normal validation FPR cap.

    Scores strictly greater than threshold are alerts; ties are conservative.
    """
    scores = np.asarray(normal_scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration requires a nonempty finite score vector")
    if not 0 <= max_fpr < 1:
        raise ValueError("max_fpr must be in [0, 1)")
    allowed = int(math.floor(max_fpr * len(scores)))
    return float(np.sort(scores)[len(scores) - allowed - 1])


def evaluate(y, scores, threshold):
    predictions = np.asarray(scores) > threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold, "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)), "recall": float(tp / (tp + fn)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "false_positive_rate": float(fp / (fp + tn)),
        "alert_rate": float((tp + fp) / len(y)), "alerts": int(tp + fp),
        "false_alerts_per_10000": float(fp / len(y) * 10000),
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
    }
