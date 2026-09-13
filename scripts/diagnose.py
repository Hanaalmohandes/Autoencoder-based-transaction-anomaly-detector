"""Reproducible post-hoc report; never changes selected models or thresholds."""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.metrics import precision_recall_curve, roc_curve

from transaction_anomaly.cli import predict_frame
from transaction_anomaly.core import Autoencoder, FEATURES, evaluate, feature_matrix, split_data
from transaction_anomaly.diagnostics import (ResidualScorers, optimal_threshold,
                                            residuals_and_latent, threshold_sweep)

OUT = Path("reports/diagnostics")


def load(path, frame):
    with (path / "preprocessing_and_baseline.pkl").open("rb") as f:
        bundle = pickle.load(f)
    indices = bundle.get("feature_indices", list(range(len(FEATURES))))
    x = bundle["scaler"].transform(feature_matrix(frame)).astype(np.float32)[:, indices]
    model = Autoencoder(len(indices), bundle.get("latent_dim", 8), bundle.get("activation", "relu"))
    model.load_state_dict(torch.load(path / "autoencoder.pt", weights_only=True))
    residual, latent = residuals_and_latent(model, x)
    return bundle, model, x, residual, latent


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    train, val, test = split_data(pd.read_csv("data/creditcard.csv"))
    fit = train[train.Time < train.Time.quantile(.8)]
    normal_fit = fit[fit.Class == 0]
    cal, selection = val[val.Time < val.Time.quantile(.5)], val[val.Time >= val.Time.quantile(.5)]
    original, improved = Path("outputs/calibrated-42"), Path("outputs/high-recall-42")
    source = "Source: OpenML 1597 · chronological test Time 145234–172792 seconds · reused test, exploratory"
    metrics, case_frames, distributions, summaries, raw_distributions = [], [], {}, {}, {}
    for name, path in [("Before: capped AE, all 29 inputs", original),
                       ("Improved: label-informed AE", improved)]:
        scored, thresholds = predict_frame(test, path)
        saved = pd.read_csv(path / "test_scores.csv")
        for kind in ["autoencoder", "isolation_forest"]:
            np.testing.assert_allclose(scored[kind + "_score"], saved[kind + "_score"], rtol=1e-6)
            np.testing.assert_array_equal(scored[kind + "_alert"], saved[kind + "_alert"])
        m = evaluate(test.Class, scored.autoencoder_score, thresholds["autoencoder"])
        metrics.append({"model": name, **m})
        if path == improved:
            metrics.append({"model": "Tuned Isolation Forest, label-informed inputs",
                            **evaluate(test.Class, scored.isolation_forest_score, thresholds["isolation_forest"])})
        bundle, model, x, residual, latent = load(path, test)
        errors = residual ** 2
        features = [FEATURES[j] for j in bundle.get("feature_indices", range(len(FEATURES)))]
        groups = np.where(test.Class.to_numpy() == 0, "Legitimate",
                          np.where(scored.autoencoder_alert, "Detected fraud", "Missed fraud"))
        distributions[name] = (scored.autoencoder_score.to_numpy(), groups, thresholds["autoencoder"])
        raw_distributions[name] = (errors.mean(1), groups)
        summaries[name] = {}
        for group in ["Legitimate", "Detected fraud", "Missed fraud"]:
            mask = groups == group
            summaries[name][group] = {
                "count": int(mask.sum()),
                "score_quantiles_10_50_90": np.quantile(scored.autoencoder_score[mask], [.1,.5,.9]).tolist(),
                "raw_mse_median": float(np.median(errors[mask].mean(1))),
                "amount_median": float(np.median(test.Amount.to_numpy()[mask])),
                "median_abs_standardized_input": float(np.median(np.abs(x[mask]))),
                "median_error_by_feature": dict(zip(features, np.median(errors[mask], axis=0).astype(float))),
            }
        for i in np.flatnonzero(test.Class.to_numpy() == 1):
            top = np.argsort(-errors[i])[:3]
            case_frames.append({"model": name, "source_row": int(test.index[i]), "Time": float(test.Time.iloc[i]),
                "Amount": float(test.Amount.iloc[i]), "anomaly_score": float(scored.autoencoder_score.iloc[i]),
                "predicted_class": int(scored.autoencoder_alert.iloc[i]), "raw_mean_reconstruction_error": float(errors[i].mean()),
                "threshold": thresholds["autoencoder"],
                "distance_from_threshold": float(scored.autoencoder_score.iloc[i] - thresholds["autoencoder"]),
                "largest_error_features": "; ".join(f"{features[j]}: {errors[i,j]:.4f}" for j in top),
                "features_at_residual_cap": int((errors[i] >= bundle["score_clip"]).sum())})
    cases = pd.DataFrame(case_frames)
    cases.to_csv(OUT / "fraud-case-analysis.csv", index=False)
    (OUT / "group-statistics.json").write_text(json.dumps(summaries, indent=2))
    pd.DataFrame(metrics).to_csv(OUT / "before-after.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    colors = ["#64748b", "#087e8b", "#b44b36"]
    for ax, (name, (scores, groups, threshold)) in zip(axes, distributions.items()):
        bins = np.linspace(0, max(scores.max(), threshold) * 1.02, 55)
        for group, color in zip(["Legitimate", "Detected fraud", "Missed fraud"], colors):
            values = scores[groups == group]
            ax.hist(values, bins=bins, weights=np.ones(len(values)) / len(values),
                    histtype="step", linewidth=1.6, label=f"{group} (n={len(values):,})", color=color)
        ax.axvline(threshold, color="black", linestyle="--", label="Frozen threshold")
        ax.set(title=name, xlabel="Anomaly score (mean capped squared standardized residual)",
               ylabel="Fraction within each group (log scale)", yscale="log")
        ax.legend(fontsize=8)
    fig.text(.02, .01, source, fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(OUT / "score-distributions.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, (name, (errors, groups)) in zip(axes, raw_distributions.items()):
        values = np.log10(np.maximum(errors, 1e-8))
        bins = np.linspace(values.min(), values.max(), 55)
        for group, color in zip(["Legitimate", "Detected fraud", "Missed fraud"], colors):
            v = values[groups == group]
            ax.hist(v, bins=bins, weights=np.ones(len(v)) / len(v), histtype="step",
                    linewidth=1.6, label=group, color=color)
        ax.set(title=name, xlabel="log10(raw mean squared standardized reconstruction error)",
               ylabel="Fraction within each group (log scale)", yscale="log")
        ax.legend(fontsize=8)
    fig.text(.02,.01,source,fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(OUT / "raw-error-distributions.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (scores, _, threshold) in distributions.items():
        fpr, tpr, _ = roc_curve(test.Class, scores)
        precision, recall, _ = precision_recall_curve(test.Class, scores)
        axes[0].plot(fpr * 100, tpr * 100, label=name)
        axes[1].plot(recall * 100, precision * 100, label=name)
    axes[0].axvline(.1, color="black", linestyle="--", label="0.1% FPR cap")
    axes[0].set(title="Test ROC near the operating limit", xlabel="False-positive rate (%)",
                ylabel="Fraud recall (%)", xlim=(0,.5), ylim=(0,100))
    axes[1].set(title="Test precision–recall", xlabel="Fraud recall (%)", ylabel="Fraud precision (%)",
                xlim=(0,100), ylim=(0,100))
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.text(.02, .01, source + " · curves are descriptive, not used to select thresholds", fontsize=7)
    fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(OUT / "roc-pr.png", dpi=160); plt.close(fig)

    # Alternative scoring uses only original model + development data.
    _, model, _, fit_r, fit_z = load(original, normal_fit)
    scorer = ResidualScorers().fit(fit_r, fit_z)
    _, _, _, cr, cz = load(original, cal)
    _, _, _, sr, sz = load(original, selection)
    cal_scores, sel_scores = scorer.score(cr, cz), scorer.score(sr, sz)
    scoring_results = []
    for name, values in cal_scores.items():
        curve = threshold_sweep(cal.Class, values)
        curve.to_csv(OUT / f"thresholds-{name}.csv", index=False)
        chosen = optimal_threshold(cal.Class, values, .001)
        scoring_results.append({"scorer": name, "calibration": chosen,
                               "selection": evaluate(selection.Class, sel_scores[name], chosen["threshold"])})
    (OUT / "alternative-scoring.json").write_text(json.dumps(scoring_results, indent=2))
    # Distribution shifts: normal rows only; KS is a descriptive effect size.
    a, b = feature_matrix(normal_fit), feature_matrix(test[test.Class == 0])
    shift = pd.DataFrame([{"feature": f, "normal_fit_vs_test_KS": float(ks_2samp(a[:,j],b[:,j]).statistic),
                           "fit_median": float(np.median(a[:,j])), "test_median": float(np.median(b[:,j]))}
                          for j,f in enumerate(FEATURES)]).sort_values("normal_fit_vs_test_KS", ascending=False)
    shift.to_csv(OUT / "normal-distribution-shift.csv", index=False)

    display = pd.DataFrame([{"Model": m["model"], "Fraud caught / 74": m["true_positives"],
        "Missed": m["false_negatives"], "Recall": f'{m["recall"]:.2%}', "Precision": f'{m["precision"]:.2%}',
        "False alerts": m["false_positives"], "FPR": f'{m["false_positive_rate"]:.4%}', "F1": f'{m["f1"]:.3f}'} for m in metrics])
    score_display = pd.DataFrame([{"Score": r["scorer"], "Selection recall": f'{r["selection"]["recall"]:.2%}',
        "Selection FPR": f'{r["selection"]["false_positive_rate"]:.4%}',
        "Meets FPR cap": r["selection"]["false_positive_rate"] <= .001} for r in scoring_results])
    development = json.loads(Path("outputs/high-recall-42/development.json").read_text())
    architecture = {}
    for row in development:
        if row["model"] != "autoencoder":
            continue
        key = (len(row["input_features"]), row["latent_dim"], row["activation"])
        m = row["selection"]
        rank = (m["false_positive_rate"] <= .001, m["recall"], m["precision"])
        if key not in architecture or rank > architecture[key][0]:
            architecture[key] = (rank, row)
    architecture_display = pd.DataFrame([{"Inputs": k[0], "Bottleneck": k[1], "Activation": k[2],
        "Selection recall": f'{r[1]["selection"]["recall"]:.2%}',
        "Selection FPR": f'{r[1]["selection"]["false_positive_rate"]:.4%}'} for k,r in architecture.items()])
    ablations = json.loads(Path("reports/training-ablations.json").read_text())
    ablation_display = pd.DataFrame([{"Training variant": a["name"], "Normal fitting rows": a["normal_training_rows"],
        "Selection recall": f'{a["chosen"]["selection"]["recall"]:.2%}',
        "Selection FPR": f'{a["chosen"]["selection"]["false_positive_rate"]:.4%}'} for a in ablations["ablations"]])
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Fraud detector diagnostic report</title><style>
    body{{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#182331;background:#fafbfc}}
    h1{{font-size:28px}}h2{{font-size:21px;margin-top:36px}}table{{border-collapse:collapse;width:100%;font-size:13px}}
    th,td{{padding:9px;text-align:left;border-bottom:1px solid #d9dfe5}}th{{background:#edf1f4;position:sticky;top:0}}
    .note{{border-left:4px solid #087e8b;padding:10px 18px;background:#edf6f6}}img{{width:100%;height:auto}}
    .scroll{{overflow:auto;max-height:550px}}a{{color:#086778}}small{{color:#536170}}</style>
    <h1>Fraud detection: 53 of 74 caught at 0.0565% FPR</h1>
    <p class="note"><b>Exploratory, reused test.</b> The 71.6% recall result exceeds the requested 70% target at the unchanged 0.1% FPR limit.
    Feature selection uses fitting-period fraud labels. Model weights use only normal rows. This is label-informed anomaly detection;
    it is not an entirely unsupervised pipeline, and it needs confirmation on new data.</p>
    <h2>Before / after and a fairly tuned baseline</h2>{display.to_html(index=False)}
    <p>Same 56,746 test transactions, 74 frauds. Inputs and thresholds were selected on development periods.
    Both models received feature-subset and hyperparameter searches. The previous result is exactly 16/74, not approximately 14/70.</p>
    <h2>What changed</h2><p>The prior score averaged 29 feature errors, including weak fraud indicators.
    The selected model uses eight historically informative inputs and caps squared residual contributions at 9.
    The architecture and threshold were chosen using separate early-stopping, calibration, and selection periods.
    The complete search and selected configuration are recorded in the accompanying JSON reports.</p>
    <h2>Where missed fraud overlaps legitimate behavior</h2><img src="score-distributions.png" alt="Anomaly score distributions by legitimate, detected fraud and missed fraud groups">
    <p>The overlap is descriptive evidence of the remaining difficulty; it is not proof that higher recall is impossible.
    Raw reconstruction errors and clipped anomaly scores are distinct. The case table records both.</p>
    <img src="raw-error-distributions.png" alt="Raw reconstruction error distributions before clipping, by outcome">
    <p>The 21 remaining missed frauds have median anomaly score 0.968, compared with 0.544 for legitimate transactions
    and 7.148 for detected frauds. Their median raw V14 squared residual is 0.318, versus 0.204 for legitimate transactions
    and 48.621 for detected frauds. Many of these cases look relatively normal in the selected feature space;
    this is not evidence that they are non-fraudulent.</p>
    <h2>Ranking and operating trade-offs</h2><img src="roc-pr.png" alt="ROC near the FPR cap and precision recall curves">
    <h2>Pure model score alternatives: development only</h2>{score_display.to_html(index=False)}
    <p>All score normalization and covariance statistics use normal fitting rows only. For each scorer, every distinct calibration threshold
    was evaluated; choose maximum recall at FPR ≤ 0.001, then maximum precision. Selection FPR can exceed calibration FPR under shift.
    These alternatives have not been evaluated or selected on test labels and do not replace the frozen improved model.</p>
    <h2>Architecture search: development results</h2>{architecture_display.to_html(index=False)}
    <p>All candidates use 24-unit hidden layers on either side of the bottleneck and a linear output. Smaller input sets use
    label-informed feature selection. Each row shows its best eligible scoring/threshold configuration. The selected network is
    8 → 24 → 2 → 24 → 8 with ReLU hidden activations. Lower normal reconstruction loss alone did not identify the strongest detector.</p>
    <h2>Training ablations: development only</h2>{ablation_display.to_html(index=False)}
    <p>L2 did not improve recall; denoising dropout reduced it. The smaller training subset with changed optimization matched recall
    with fewer development false positives, but was not tested on the reused test set or promoted over the frozen model.
    That combined ablation cannot isolate the effect of subset size from learning rate, batch size, or patience.</p>
    <h2>Normal-data shift: five largest KS distances</h2>{shift.head().to_html(index=False,float_format=lambda v:f"{v:.4f}")}
    <p>KS distance is a distribution difference, not a claim of causal drift. Both periods contain legitimate transactions only.</p>
    <h2>Every fraud transaction: before and after</h2>
    <p>Positive distance means above threshold. Source row is the zero-based CSV data-row index.
    Largest features refer to raw squared residuals in each model's standardized input space.</p>
    <p><a href="fraud-case-analysis.csv">Download all 148 model/case rows</a> · <a href="group-statistics.json">Group-level score and feature statistics</a>
    · <a href="alternative-scoring.json">Score comparison details</a></p>
    <div class="scroll">{cases.to_html(index=False,float_format=lambda v:f"{v:.5f}")}</div>
    <p><small>{source}. Repeated model development makes reused-test results optimistic; no independent holdout exists in this project.</small></p></html>'''
    (OUT / "index.html").write_text(html)
    print(json.dumps({"metrics": metrics, "group_statistics": summaries,
                      "score_alternatives": scoring_results}, indent=2))


if __name__ == "__main__":
    main()
