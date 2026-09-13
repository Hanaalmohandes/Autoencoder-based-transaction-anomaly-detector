# Autoencoder-based transaction anomaly detector

A PyTorch autoencoder learns the structure of legitimate transactions and flags high reconstruction errors. An Isolation Forest provides a normal-only baseline. The experiment asks whether the autoencoder can improve fraud recall while staying within a configurable false-positive budget.

This is an experiment, not a claim that deep learning will outperform the baseline. No CIB or other private customer data is used.

## Latest result: 71.6% recall at 0.0565% false positives

The frozen label-informed autoencoder detects **53/74 frauds**, compared with **16/74** for the previous capped autoencoder and **40/74** for the newly tuned Isolation Forest. It generates **32 false alerts**, compared with 38 previously and 34 for the new baseline. Recall is **71.62%**, precision **62.35%**, F1 **0.667**, and FPR **0.0565%**, satisfying the requested **70% recall / 0.1% FPR** operating target on this reused test set.

**This remains exploratory evidence, not an untouched holdout.** Feature ranking uses fraud labels from the fitting period. Network and forest fitting use normal transactions only; the complete pipeline is label-informed anomaly detection, not purely unsupervised learning. The original and pure reconstruction-based variants remain available below. Historical adaptive reuse of the test set and small fraud counts may inflate apparent performance.

The main changes are an eight-input feature subset, a two-unit bottleneck, and capped reconstruction scoring. Inputs are V14, V17, V12, V16, V11, V18, V4, and V3, ranked by fitting-period average precision of absolute standardized feature values. Architecture: **8 → 24 → 2 → 24 → 8**, ReLU hidden activations, linear output. The anomaly score averages squared residuals capped at 9 per feature. The threshold **4.922752380371094** was derived from normal calibration scores at a stricter 0.05% calibration budget, then selected subject to a 0.1% selection FPR cap. No test labels set the threshold.

Calibration: **15/24 frauds**, 62.50% recall, 14 false positives (0.0494% FPR). Model-selection period: **26/33 frauds**, 78.79% recall, 26 false positives (0.0917% FPR). Test: **53/74 frauds**, 32 false positives. The weaker calibration-period recall illustrates time variation; 70% is not guaranteed on future periods.

Open [the diagnostic report](reports/diagnostics/index.html) for before/after metrics, score and raw reconstruction-error distributions, ROC/PR plots, architecture comparisons, training ablations, normal-distribution shifts, and all 74 fraud cases before and after. [The case CSV](reports/diagnostics/fraud-case-analysis.csv) includes source row, anomaly score, predicted class, raw reconstruction MSE, largest-error features, and distance from threshold.

To reproduce, use Python 3.12 and the installed project environment:

```bash
python -m pip install -e '.[dev,analysis]'
PYTHONPATH=src python scripts/high_recall.py --output outputs/new-high-recall-run
PYTHONPATH=src python scripts/training_ablations.py
MPLCONFIGDIR=/tmp/transaction-matplotlib PYTHONPATH=src python scripts/diagnose.py
```

The diagnostic and ablation scripts reference the preserved `outputs/high-recall-42/` and `outputs/calibrated-42/` runs. `high_recall.py` records its protocol before fitting and all development candidates before test scoring. It only scores test data if development reaches the recall/FPR target. It tests eight architecture/input configurations and six forest configurations, with scoring and calibration variants (120 total candidates). All selected input indices, activations, model weights, scalers, and thresholds are serialized.

Evaluate genuinely fresh labeled data **without retraining, retuning, or recalibrating**:

```bash
PYTHONPATH=src python -m transaction_anomaly.cli evaluate \
  --data data/fresh_holdout.csv --model-dir outputs/high-recall-42 \
  --output outputs/fresh-evaluation.json
```

For unlabeled transactions, use `score` with the same model directory. The caller must establish that a fresh holdout is independent; the command cannot infer that from a file name.

### Diagnosis and additional experiments

- All feature scaling and model fitting use normal fitting rows only. No fraud examples enter either detector's parameter fitting. Fraud labels influence input selection and development model selection, explicitly.
- The all-feature reconstruction score diluted relevant errors and was sensitive to heavy-tailed normal behavior. The feature/architecture search improved development separation; simply minimizing normal reconstruction MSE did not select the best detector.
- Twelve score alternatives were tested **on development data only**, including standardized residuals, robust z normalization, percentile-tail errors, top-k/max errors, residual/latent Mahalanobis scores, and combinations. Threshold sweeps enumerate all distinct decisions and maximize calibration recall subject to FPR ≤ 0.001, with precision as a tie-break. Exact curves are generated locally under `reports/diagnostics/thresholds-*.csv`. Their independent selection metrics are in `alternative-scoring.json`; no alternative was promoted based on test results.
- L2 regularization matched development recall; denoising dropout reduced it. A 25% normal subset with lower learning rate, smaller batches, and longer patience matched recall with fewer development false positives. This is a combined ablation, so improvements cannot be attributed to subset size alone. It is retained for new-holdout work, not substituted into the reported result.
- The 21 remaining missed frauds have median anomaly score **0.968**, versus **0.544** for normal transactions and **7.148** for detected frauds. Median raw V14 squared residual is **0.318** for misses, **0.204** for legitimate transactions, and **48.621** for detections. The missed group is much closer to legitimate behavior in these inputs; the available analysis does not prove higher recall impossible.
- Normal feature distributions shift across time (for example V3 KS distance 0.535 and V1 0.421 between fitting and test). These are post-hoc descriptions, not causal diagnoses or inputs to tuning.
- No logistic-regression score classifier was added: the retained deployed score remains a direct reconstruction-error function, with label-informed feature selection clearly disclosed. Batch normalization, deeper networks, and L1 penalties were not exhaustively searched; larger searches risk overfitting the 33-fraud selection set.
- Existing scoring functionality is preserved. Tests cover exact threshold decisions/ties, F1, finite score normalizers, selected-feature/tanh serialization, separate model scalers, and frozen holdout evaluation. Full-test saved-score/alert parity is checked by the report script.

## Improved benchmark: target observed on the reused test set

The revised autoencoder caught **16 of 74 frauds (21.62% recall)** versus **3 of 74 (4.05%)** for the tuned Isolation Forest. Autoencoder false-positive rate was **0.0671%**, below the unchanged 0.1% budget; baseline FPR was 0.0600%. The autoencoder generated 54 alerts, including 38 false alerts (29.63% precision), versus 37 alerts, including 34 false alerts (8.11% precision), for the baseline. That is 13 additional fraud detections for four additional false alerts on 56,746 transactions.

**This is exploratory evidence on an already inspected test set, not a fresh holdout result.** The original run and two unsuccessful revisions were retained. No claim of production readiness or reliable out-of-sample superiority is justified by this experiment. The autoencoder still missed 58 frauds, and the 0.1% budget remains an illustrative assumption about review capacity.

The selected change caps each feature's squared reconstruction residual at 4 before averaging. This limits how much one extreme feature can dominate the anomaly score. Standard scaling and an eight-dimensional bottleneck won the development search. A stricter calibration FPR of 0.075% provided margin for changes between time periods, while the selection/test acceptance cap remained 0.1%.

For the revised experiment, the original training period is divided chronologically into fitting (80%) and early stopping (20%). The original validation period is divided into independent calibration and model-selection halves. Preprocessing fits only legitimate fitting rows. Both detectors use the same fitting rows and get a development search. Autoencoders compare two preprocessors, three bottleneck sizes, four residual caps, and three calibration budgets (72 combinations); Isolation Forest compares two preprocessors, two sample sizes, and the same three budgets (12 combinations). Select the highest recall among models within the selection FPR cap, breaking ties by average precision. If none satisfy the cap, the search reports a best-effort candidate; inspect its selection metrics before interpreting it as eligible. Model-selection labels are used, so the overall workflow is semi-supervised.

The original test period remains unchanged. Each revision evaluated only its development-selected models on that test, but choosing further revisions after observing failures creates adaptive reuse. The selection partition contains just 33 frauds; 84 candidates can overfit that small sample. Validate the frozen method on new data before making a durable performance claim.

Reproduce the final search from the repository root (use an unused output directory):

```bash
PYTHONPATH=src .venv/bin/python scripts/improve.py --output outputs/reproduced-improvement
PYTHONPATH=src .venv/bin/python -m transaction_anomaly.cli score \
  --data data/new_transactions.csv --model-dir outputs/calibrated-42 \
  --output outputs/new_scores.csv
```

The first command writes a protocol before training, all development scores before test evaluation, metrics, test scores, and model artifacts. `reports/improved-benchmark.json` preserves final results and `reports/development-search.json` preserves every final candidate. `reports/scaling-attempt.json` and `reports/capping-attempt.json` preserve unsuccessful revisions. The standard `train` command below continues to reproduce the original unmodified baseline experiment.

## Initial benchmark result

The first full public-data run used seed 42, 30 epochs, the chronological split, and a 0.1% validation false-positive budget. After removing 1,081 exact duplicate rows, the test set contained 56,746 transactions including 74 frauds. **Neither detector caught any of those 74 frauds at its calibrated threshold.** The autoencoder produced 21 false alerts (0.0371% FPR); Isolation Forest produced 34 (0.0600% FPR). The target was not achieved.

The autoencoder's test average precision was 0.1591 versus 0.0411 for Isolation Forest, indicating stronger ranking on this split despite the unsuccessful alert operating point. This does not justify a higher-recall claim. The exact metrics and data checksum are preserved in `reports/initial-benchmark.json`; locally generated weights and transaction scores are in `outputs/run-42/`.

Next experiments should prespecify feasible review budgets and tune on development data, then use a fresh evaluation period. Do not relax thresholds after inspecting this test set and present the result as an untouched benchmark.

## Setup and run

Use Python 3.10 or newer (Python 3.12 recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
transaction-anomaly download
transaction-anomaly train --max-fpr 0.001 --epochs 30 --output outputs/run-42
pytest -q
```

`requirements-lock.txt` records the exact dependency versions used in this workspace (Python 3.12, macOS ARM). To recreate those versions, install it with `python -m pip install -r requirements-lock.txt` before installing the project. Availability of matching wheels depends on your platform.

The download command fetches the [OpenML creditcard dataset, ID 1597](https://www.openml.org/d/1597). The original [ULB / Worldline dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) contains anonymized transactions and a binary fraud label. Alternatively, download `creditcard.csv` from the original source and place it in `data/`. Review the source's license and attribution terms before redistribution. Raw data and generated model artifacts are ignored by Git.

## Experiment design

1. Validate finite numeric data and remove exact duplicate transactions.
2. Split chronologically at approximately 60% / 20% / 20%, keeping equal timestamps together. All three splits must contain both classes. An optional `--split stratified` gives a seeded random comparison, but does not test future-period generalization.
3. Use V1–V28 and `log1p(Amount)`. Exclude `Time` from model inputs because it is elapsed time rather than a stable transaction attribute. Fit standardization only on normal training rows.
4. Train a 29 → 24 → 8 → 24 → 29 autoencoder with ReLU hidden layers, a linear output, mean squared reconstruction loss, and Adam. Early stopping uses normal validation reconstruction loss with patience 5 and restores the best weights. Train a 300-tree Isolation Forest on the same normal training rows.
5. Independently calibrate each model's threshold on normal validation scores. Flag scores **strictly greater** than the threshold. Ties are conservative; the empirical validation false-positive rate never exceeds the requested cap.
6. Evaluate the frozen models and thresholds once on the held-out test set. Report recall, precision, average precision, ROC-AUC, false-positive rate, confusion counts, alert rate, and false alerts per 10,000 transactions.

Labels identify normal training/calibration examples and evaluate performance; this is a **semi-supervised novelty-detection setup** using unsupervised model objectives. Both methods benefit from the same normal-only training assumption. No oversampling or supervised fraud-class optimization is applied.

The default 0.1% false-positive budget is an illustrative operating point, not a validated review-team capacity. Actual workload also depends on transaction volume and fraud prevalence. A validation FPR cap is not a guarantee on future data; test FPR can exceed the cap under distribution shift. The report makes this explicit for both models.

## Outputs

Each run writes to a new, empty directory:

- `metrics.json`: settings, data checksum, dependency versions, split counts, both models' validation/test metrics, and observed target status.
- `training_history.csv`: training and normal-validation reconstruction loss per epoch.
- `test_scores.csv`: original zero-based CSV data-row index, labels, anomaly scores, and alert flags for both models.
- `autoencoder.pt`: restored best model weights.
- `preprocessing_and_baseline.pkl`: fitted scaler, Isolation Forest, feature order, and thresholds.

Higher scores mean more anomalous for both detectors. Scores are not fraud probabilities. The target is marked observed only when the autoencoder's test recall exceeds the baseline and its test FPR stays within the specified budget. This describes one split; it does not establish a statistically reliable improvement.

## Score new transactions

Use a CSV containing V1–V28 and Amount; Class and Time are optional for inference.

```bash
transaction-anomaly score --data data/new_transactions.csv \
  --model-dir outputs/run-42 --output outputs/new_scores.csv
```

Load only your own trusted artifacts: the preprocessing/baseline pickle can execute code. Keep the training environment for inference; scikit-learn artifacts are not portable across arbitrary versions.

## Interpretation and limitations

- Inspect precision and alert counts alongside recall. High recall with excessive false alerts may be operationally unusable.
- The public benchmark is anonymized and covers a short historical period. It cannot establish performance on a bank's current transaction stream.
- Known-normal training assumes reliable labels; real-world contamination may reduce detection quality.
- Validation supports early stopping and threshold calibration. A production study should add independent calibration periods and test stability across time windows.
- After viewing test outcomes, do not repeatedly tune against the same test split. Use a fresh holdout for subsequent claims. Compare multiple prespecified seeds and uncertainty intervals before claiming a durable improvement.
- Automated tests use synthetic fixtures to verify correctness, not to demonstrate fraud-detection performance.

## Code

`src/transaction_anomaly/core.py` contains preprocessing, training, scoring, threshold calibration, and metrics. `cli.py` provides download/train/score commands. `tests/test_pipeline.py` checks split separation, training-only scaling, threshold behavior, known metrics, and saved-model inference parity.
