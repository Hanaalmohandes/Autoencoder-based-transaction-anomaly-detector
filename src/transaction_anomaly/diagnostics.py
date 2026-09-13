"""Post-hoc diagnostics and development-only score/threshold experiments."""
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf


def threshold_sweep(y, scores):
    """All distinct strict-greater-than decisions, computed in O(n log n)."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError("Expected equal nonempty label and score vectors")
    if not np.isfinite(scores).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected finite scores and both binary classes")
    order = np.argsort(-scores, kind="stable")
    s, labels = scores[order], y[order]
    # At each group start, only the preceding (strictly larger) scores alert.
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    tp = np.r_[0, np.cumsum(labels)][starts]
    fp = starts - tp
    thresholds = s[starts]
    tp, fp = np.r_[tp, y.sum()], np.r_[fp, len(y) - y.sum()]
    thresholds = np.r_[thresholds, np.nextafter(s[-1], -np.inf)]
    fn, tn = y.sum() - tp, len(y) - y.sum() - fp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=tp + fp > 0)
    return pd.DataFrame({"threshold": thresholds, "recall": tp / y.sum(),
        "precision": precision, "f1": 2 * tp / (2 * tp + fp + fn),
        "true_positives": tp, "false_negatives": fn, "false_positives": fp,
        "true_negatives": tn, "false_positive_rate": fp / (len(y) - y.sum())})


def optimal_threshold(y, scores, max_fpr=.001):
    """Maximize empirical recall under an FPR constraint; break ties by precision."""
    if not 0 <= max_fpr < 1:
        raise ValueError("max_fpr must be in [0, 1)")
    curve = threshold_sweep(y, scores)
    feasible = curve[curve.false_positive_rate <= max_fpr]
    return feasible.sort_values(["recall", "precision", "threshold"],
                                ascending=[False, False, False]).iloc[0].to_dict()


def residuals_and_latent(model, x, batch_size=4096):
    model.eval()
    residuals, latent = [], []
    with torch.no_grad():
        for batch in torch.from_numpy(x).split(batch_size):
            residuals.append((model(batch) - batch).numpy())
            latent.append(model.network[:4](batch).numpy())
    return np.concatenate(residuals), np.concatenate(latent)


class ResidualScorers:
    """All reference statistics fit on normal fitting residuals only."""
    def fit(self, residuals, latent):
        e = residuals.astype(float) ** 2
        self.mean = np.maximum(e.mean(0), 1e-6)
        self.median = np.median(e, axis=0)
        self.mad = np.maximum(1.4826 * np.median(np.abs(e - self.median), axis=0), 1e-6)
        self.sorted_errors = np.sort(e, axis=0)
        self.residual_cov = LedoitWolf().fit(residuals)
        self.latent_cov = LedoitWolf().fit(latent)
        return self

    def score(self, residuals, latent):
        e = residuals.astype(float) ** 2
        normalized = e / self.mean
        robust = np.maximum((e - self.median) / self.mad, 0)
        n = len(self.sorted_errors)
        tail = np.stack([(n + 1 - np.searchsorted(self.sorted_errors[:, j], e[:, j], side="right"))
                         / (n + 1) for j in range(e.shape[1])], axis=1)
        percentile = -np.log(tail)
        mean, largest = e.mean(1), e.max(1)
        residual_maha = self.residual_cov.mahalanobis(residuals) / residuals.shape[1]
        latent_maha = self.latent_cov.mahalanobis(latent) / latent.shape[1]
        return {"mean_mse": mean, "capped_mse_4": np.minimum(e, 4).mean(1),
            "standardized_residual": normalized.mean(1),
            "robust_z_log": np.log1p(robust).mean(1),
            "percentile_tail": percentile.mean(1),
            "top3_percentile_tail": np.sort(percentile, axis=1)[:, -min(3, e.shape[1]):].mean(1),
            "top3_mse": np.sort(e, axis=1)[:, -min(3, e.shape[1]):].mean(1),
            "max_mse": largest, "mean_max_blend": .5 * (mean + largest),
            "residual_mahalanobis": residual_maha, "latent_mahalanobis": latent_maha,
            "residual_latent_blend": .5 * (residual_maha + latent_maha)}
