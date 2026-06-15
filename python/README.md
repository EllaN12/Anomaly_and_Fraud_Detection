# Credit Card Fraud Detection — Python Pipeline

End-to-end fraud analytics on the [ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), covering anomaly detection, class-imbalance handling, objective feature selection, and temporal model validation.

---

## Dataset

`creditcard.csv` — 284,807 European credit card transactions over 48 hours (September 2013).

| Attribute | Value |
|-----------|-------|
| Total transactions | 284,807 |
| Fraud cases | 492 |
| Fraud rate | 0.172% |
| Imbalance ratio | 577 : 1 |
| Features | V1–V28 (PCA-transformed), Amount, Time |
| Label | `Class` (1 = fraud, 0 = legitimate) |

V1–V28 are anonymised PCA components; original features are not available for confidentiality reasons.

---

## Pipeline

Seven scripts run in sequence. Each writes its outputs to `outputs/`.

```
python run_all.py
```

| Script | Purpose |
|--------|---------|
| `00_data_prep.py` | Load creditcard.csv, class imbalance diagnostics, Amount distribution by class |
| `01_experiment_1_point_anomalies.py` | Controlled point-anomaly experiment — single outlier in synthetic Gaussian cluster |
| `02_experiment_2_timeseries_anomalies.py` | Time-series anomaly experiment — IsolationForest + STL decomposition on synthetic linear trend with injected spikes |
| `03_experiment_3_cluster_anomalies.py` | Cluster anomaly experiment — satellite cluster flagged as collective anomaly |
| `04_fraud_anomaly_detection.py` | IsolationForest on creditcard.csv with corrected contamination, F1-optimal threshold calibration, SMOTE + Random Forest supervised comparison |
| `05_feature_discrimination.py` | Objective SCM suitability ranking of all 28 PCA components — determines the best donor-stratification feature programmatically |
| `06_temporal_validation.py` | Walk-forward temporal validation across 8-hour blocks, multi-feature PSI drift monitoring |

---

## Key Design Decisions

### Class imbalance

Standard `contamination='auto'` in sklearn maps to 10%, implying ~28,000 anomalies. The true fraud rate is 0.172%. Every IsolationForest in this pipeline uses `contamination=fraud_rate` to calibrate the decision boundary to the actual minority fraction.

PR AUC replaces ROC AUC as the primary metric. ROC AUC reaches 0.955 for an unsupervised model because the 284K-strong majority class dominates the false-positive-rate denominator — it flatters any detector. PR AUC of 0.189 against a no-skill baseline of 0.0017 is the honest measure.

Thresholds are calibrated on a held-out validation set by maximising F1, not set to a fixed percentile.

### Feature discrimination (script 05)

No PCA component is pre-selected by assumption. All V1–V28 are scored by a two-factor **SCM Suitability Score**:

```
scm_score = disc_power × |std_separation|

disc_power     = |AUROC − 0.5|
std_separation = (fraud_mean − nf_mean) / nf_std
```

`disc_power` measures how well the feature separates classes. `std_separation` measures how far the fraud cohort sits from the non-fraud centre in non-fraud standard deviation units — a large value ensures the decile-ranked donor pool spans a true fraud-like to typical spectrum rather than clustering near the centre.

Only features where the fraud cohort sits below the non-fraud mean are eligible, so decile groups run from most-fraud-like to most-typical as values increase.

**Result:** V14 is selected (scm_score = 3.497, disc_power = 0.391, |sep| = 8.94σ). V12 ranks second (scm_score = 2.897). The winner is determined on every run — the script contains no hardcoded feature names.

### Temporal validation (script 06)

Random train/test splits ignore the temporal ordering of transactions and allow future data to leak into training. Walk-forward cross-validation respects the time axis: the model always trains on the past and is tested on the future.

The 48-hour window is divided into 6 × 8-hour blocks (B0–B5), producing 4 expanding-window folds:

```
Fold 1: Train=B0              | Val=B1 | Test=B2
Fold 2: Train=B0+B1           | Val=B2 | Test=B3
Fold 3: Train=B0+B1+B2        | Val=B3 | Test=B4
Fold 4: Train=B0+B1+B2+B3     | Val=B4 | Test=B5
```

Both IsolationForest and class-balanced Random Forest are evaluated per fold. A conventional random 60/20/20 stratified split serves as a methodology baseline.

### Population Stability Index (PSI)

PSI monitors whether the feature distribution seen at training time has drifted by the time the model scores a new block. PSI is computed across the **top-N most discriminating features** ranked by `disc_power` from script 05, sourced from `feature_discrimination.json`. This ensures monitoring effort is concentrated on the features the model relies on most to separate fraud from non-fraud.

Industry thresholds applied per feature per fold:

| PSI | Signal |
|-----|--------|
| < 0.10 | Stable — no action |
| 0.10 – 0.20 | Caution — increase monitoring frequency |
| > 0.20 | Retrain signal |

---

## Results Summary

### Script 04 — Fraud Detection

| Method | PR AUC | F1 | Precision | Recall |
|--------|--------|----|-----------|--------|
| IsolationForest (F1-optimal threshold) | 0.189 | 0.311 | 0.294 | 0.331 |
| RF balanced (5-fold stratified CV) | — | 0.898 | — | 0.829 |
| SMOTE + RF balanced | — | 0.907 | — | 0.876 |

### Script 05 — Feature Discrimination (top 5 by SCM score)

| Rank | Feature | disc_power | \|std_sep\| | scm_score | Direction |
|------|---------|------------|-------------|-----------|-----------|
| 1 | V14 | 0.391 | 8.94 | 3.497 | fraud < non-fraud |
| 2 | V12 | 0.337 | 8.60 | 2.897 | fraud < non-fraud |
| 3 | V17 | 0.310 | 8.86 | 2.746 | fraud < non-fraud |
| 4 | V10 | 0.271 | 7.98 | 2.163 | fraud < non-fraud |
| 5 | V16 | 0.226 | 6.85 | 1.547 | fraud < non-fraud |

### Script 06 — Temporal Validation

| Method | Split | Mean PR AUC | Std PR AUC |
|--------|-------|-------------|------------|
| IsolationForest | Random 60/20/20 | 0.189 | — |
| IsolationForest | Temporal (4 folds) | ~0.22 | ~0.23 |
| RF balanced | Random 60/20/20 | ~0.87 | — |
| RF balanced | Temporal (4 folds) | ~0.78 | ~0.03 |

IsolationForest is highly unstable under temporal shifting (PR AUC range 0.018–0.585 across folds). RF balanced with `class_weight='balanced'` maintains consistent performance (range 0.743–0.818, std ~0.03).

---

## Outputs

All generated files land in `outputs/`. Key files by script:

| File | Script | Description |
|------|--------|-------------|
| `amount_by_class_density.png` | 00 | Fraud vs non-fraud Amount distributions |
| `exp1_point_anomalies.png` | 01 | Point anomaly isolation experiment |
| `exp2_iforest.png` | 02 | Time-series IsolationForest anomaly flags |
| `exp3_cluster_anomalies.png` | 03 | Satellite cluster anomaly detection |
| `fraud_roc.png` | 04 | ROC curve (AUC 0.955) |
| `fraud_pr.png` | 04 | PR curve (AUC 0.189) |
| `fraud_threshold_compare.png` | 04 | Three operating thresholds compared |
| `fraud_anomaly_scatter.png` | 04 | Feature scatter — IsoForest flags vs true labels |
| `fraud_metrics.json` | 04 | Full metrics for all threshold strategies |
| `feature_scm_suitability.png` | 05 | All 28 V features ranked by SCM suitability score |
| `feature_discrimination_ranked.png` | 05 | Ranked by raw disc_power only |
| `feature_discrimination_densities.png` | 05 | KDE density grid — top 8 features |
| `feature_discrimination.json` | 05 | Full scores for all V features (read by script 06) |
| `temporal_metrics_by_fold.png` | 06 | PR AUC and F1 by fold for both models |
| `temporal_vs_random_comparison.png` | 06 | Random split vs temporal methodology |
| `temporal_psi_drift.png` | 06 | PSI heatmap (top features × folds) + aggregate lines |
| `temporal_prec_recall_by_fold.png` | 06 | Precision and Recall breakdown by fold |
| `temporal_fold_metrics.csv` | 06 | Per-fold metrics table |
| `temporal_validation_summary.json` | 06 | Aggregate summary with PSI per feature per fold |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline end-to-end
python run_all.py

# Or run individual scripts
python 05_feature_discrimination.py   # writes feature_discrimination.json
python 06_temporal_validation.py      # reads feature_discrimination.json if present
```

Script 06 reads `outputs/feature_discrimination.json` written by script 05 to determine which features to monitor for PSI. If the file is not present it computes `disc_power` inline from the CSV as a fallback, so scripts can be run independently.

---

## Requirements

```
scikit-learn
pandas
numpy
matplotlib
scipy
```

See `requirements.txt` for pinned versions.

---

## Portfolio Presentation

`credit_card_fraud_detection.pptx` in the project root — 12-slide deck covering the full pipeline, key design choices, model results, feature discrimination analysis, temporal validation findings, and PSI drift monitoring. Intended for a mixed business and technical audience.
