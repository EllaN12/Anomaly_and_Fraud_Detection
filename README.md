# Credit Card Fraud Detection — Anomaly Detection Portfolio Project

## 1. Overview

This project investigates whether fraudulent credit card transactions can be identified from real transaction data, using both unlabeled (unsupervised) and labeled (supervised) approaches, and whether any resulting model would hold up over time in production.

**Business question:** Given a stream of transactions where most are never reviewed or labeled, can we (a) detect fraud patterns without labels, (b) identify which signals reliably indicate fraud when labels do exist, and (c) determine whether a fraud-detection model trained today would still be trustworthy weeks or months later?

To answer this, the project runs seven stages against 284,807 real transactions:

1. **Pattern detection without labels** — Isolation Forest is applied as if no `Class` labels existed, to test how much fraud signal can be recovered from structure alone (anomaly/outlier behavior) rather than supervision.
2. **Label-driven signal discovery** — once labels are reintroduced, an objective scoring method ranks which of the 28 anonymized features actually separate fraud from non-fraud, rather than assuming which ones matter.
3. **Viability and longevity assessment** — a walk-forward temporal validation simulates deploying the model and re-testing it on later time windows, with drift monitoring to flag when the model's reliability degrades.

**Result in brief:** unsupervised detection recovers a meaningful but limited share of fraud (F1 = 0.311) and is **temporally unstable** — its accuracy swings considerably as fraud patterns shift over time. Supervised detection, once labels are available, is both far more accurate (F1 = 0.905) and far more stable over time, making it the more viable choice for sustained production use. Full results are in [Key Findings](#5-key-findings).

---

## 2. Data Source

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
**File:** `Raw_Data/creditcard.csv` (150.8 MB) — **not included in this repo** (exceeds GitHub's 100 MB limit). Download from Kaggle and place in `Raw_Data/` before running the scripts. See [Raw_Data/README.md](Raw_Data/README.md).

| Attribute | Value |
|-----------|-------|
| Transactions | 284,807 |
| Features | 30 PCA components (V1–V28) + Time + Amount |
| Fraud cases | 492 (0.173%) |
| Imbalance ratio | 577 : 1 |

All V1–V28 features are anonymized PCA projections of the original transaction attributes (provided this way by the data source to protect cardholder privacy). `Time` is seconds elapsed from the first transaction in the dataset (covers a 48-hour window). `Class` is the binary label (1 = fraud, 0 = legitimate).

---

## 3. Methods and Tools

| Method | Purpose |
|--------|---------|
| Isolation Forest (unsupervised) | Detects anomalies/outliers without using fraud labels |
| Random Forest (supervised, class-weighted) | Learns fraud patterns directly from labeled data |
| STL decomposition | Separates trend/seasonality from residual anomalies in time-series data |
| SCM suitability scoring | Objectively ranks features by discriminatory power, rather than relying on intuition |
| Walk-forward (expanding window) cross-validation | Tests model stability across sequential time blocks instead of random shuffles |
| Population Stability Index (PSI) | Quantifies feature drift between time periods to signal when retraining is needed |
| Precision/Recall/F1/MCC, ROC AUC, PR AUC | Evaluate performance under severe class imbalance (ROC AUC alone is misleading at 577:1) |

| Library | Purpose |
|---------|---------|
| pandas, numpy | Data loading and manipulation |
| scikit-learn | Isolation Forest, Random Forest, metrics |
| statsmodels | STL time-series decomposition |
| matplotlib | All visualizations |
| imbalanced-learn *(optional)* | SMOTE oversampling |

---

## 4. ML Models Used

- **Isolation Forest** — unsupervised anomaly detector, used both on synthetic data (to validate the method) and on the real fraud dataset (calibrated to the true fraud rate via the `contamination` parameter).
- **Random Forest with `class_weight='balanced'`** — supervised classifier used as the labeled-data baseline, evaluated against Isolation Forest on the same transactions.

---

## 5. Key Findings

1. **Imbalance is the primary challenge.** The 577:1 fraud-to-legitimate ratio invalidates accuracy and ROC AUC as standalone metrics. Calibrating Isolation Forest's `contamination` parameter to the true fraud rate (0.001727, instead of sklearn's default 10%) and selecting the detection threshold by F1 are the critical fixes for usable unsupervised detection.

2. **Unsupervised detects; supervised catches.** Isolation Forest achieves F1 = 0.311 and PR AUC = 0.189 with no labels at all. Random Forest with `class_weight='balanced'` reaches F1 = 0.905 and PR AUC = 0.916 — roughly a 3× improvement once labeled data is available.

   | Method | F1 | MCC | Recall | PR AUC |
   |--------|----|-----|--------|--------|
   | Isolation Forest | 0.311 | 0.311 | 0.331 | 0.189 |
   | RF (class_weight='balanced') | **0.905** | **0.903** | **0.854** | **0.916** |

3. **Feature selection must be objective, not assumed.** All 28 anonymized PCA components are scored by an SCM suitability metric (`scm_score = disc_power × |std_separation|`). **V14** ranks highest (score 3.497), followed by V12, V17, V10, and V3 — all five separate fraud from non-fraud in the same direction (fraud values run lower than non-fraud).

4. **Temporal validation exposes model fragility.** Walk-forward cross-validation across four expanding time windows shows Isolation Forest's PR AUC varies by ±0.273 (ranging 0.018–0.585 across folds) — effectively unreliable across time, because shifting fraud rates between time blocks break its contamination calibration. Random Forest holds steady at PR AUC 0.783 ± 0.034, making it the more viable choice for sustained deployment.

   | Method | PR AUC (avg ± std) | F1 (avg ± std) |
   |--------|--------------------|----------------|
   | Isolation Forest — temporal | 0.176 ± 0.273 | 0.199 ± 0.277 |
   | Isolation Forest — random split | 0.131 | 0.197 |
   | RF balanced — temporal | **0.783 ± 0.034** | **0.810 ± 0.026** |
   | RF balanced — random split | 0.809 | 0.827 |

5. **Drift monitoring provides an early retrain signal.** Tracking PSI on the top-5 discriminating features (V14, V4, V12, V11, V10) flagged significant drift in V12 during the first time fold (PSI = 2.93, well above the 0.20 "retrain" threshold), with drift stabilizing as the training window expanded in later folds.

---

## 6. Installation and Setup

```bash
pip install -r requirements.txt
# Optional — enables SMOTE oversampling variant
pip install imbalanced-learn
```

Download `creditcard.csv` from Kaggle (see [Data Source](#2-data-source)) and place it in `Raw_Data/`.

### Run everything

```bash
python3 run_all.py
```

Or run scripts individually in order (00 → 06). All outputs are written to `outputs/`.

---

## 7. Project Structure

```
.
├── Raw_Data/
│   └── creditcard.csv
├── outputs/              # All generated plots, CSVs, and JSON metrics
├── 00_data_prep.py
├── 01_experiment_1_point_anomalies.py
├── 02_experiment_2_timeseries_anomalies.py
├── 03_experiment_3_cluster_anomalies.py
├── 04_fraud_anomaly_detection.py
├── 05_feature_discrimination.py
├── 06_temporal_validation.py
├── run_all.py
├── requirements.txt
└── credit_card_fraud_detection.pptx
```

---

## 8. Key Components

### 00 — Data Preparation (`00_data_prep.py`)
Loads `creditcard.csv`, reports class imbalance, and generates the transaction amount density plot split by class.
**Key output:** `outputs/amount_by_class_density.png`

### 01 — Point Anomaly Experiment (`01_experiment_1_point_anomalies.py`)
Synthetic 2-D Gaussian cluster with a single injected outlier, used to validate that Isolation Forest correctly isolates a lone deviant before applying it to real fraud data.
**Output:** `outputs/exp1_point_anomalies.{csv,png}`

### 02 — Time-Series Anomaly Experiment (`02_experiment_2_timeseries_anomalies.py`)
Linear trend with two injected spikes, comparing Isolation Forest (9 flags) against STL decomposition + IQR (3 flags, precisely catching the true spikes).
**Outputs:** `outputs/exp2_timeseries_anomalies.csv`, `outputs/exp2_iforest.png`, `outputs/exp2_stl_decomposition.png`

### 03 — Cluster Anomaly Experiment (`03_experiment_3_cluster_anomalies.py`)
Base cluster at (50, 50) plus a satellite cluster at (80, 80); Isolation Forest correctly flags the remote satellite group as collective anomalies (7 of 35 points).
**Output:** `outputs/exp3_cluster_anomalies.{csv,png}`

### 04 — Fraud Anomaly Detection (`04_fraud_anomaly_detection.py`)
Core script. Fits Isolation Forest on the full real dataset, calibrates `contamination` to the true fraud rate, compares three detection thresholds, and benchmarks against a supervised Random Forest baseline.

| Threshold | Flags | Precision | Recall | F1 |
|-----------|-------|-----------|--------|----|
| 99th-pct | 2,849 | 0.109 | 0.632 | 0.186 |
| Fraud-rate calibrated | 492 | 0.295 | 0.295 | 0.295 |
| F1-optimal | 555 | 0.294 | 0.331 | **0.311** |

**Outputs:** `outputs/fraud_metrics.json`, `outputs/fraud_roc.png`, `outputs/fraud_pr.png`, `outputs/fraud_threshold_compare.png`, `outputs/fraud_anomaly_scatter.png`, `outputs/fraud_scores.csv`

### 05 — Feature Discrimination (`05_feature_discrimination.py`)
Ranks all 28 PCA components using the SCM suitability score (`disc_power × |std_separation|`), enabling data-driven donor-feature selection without hardcoding.

| Rank | Feature | Direction | disc_power | \|sep\| σ | scm_score |
|------|---------|-----------|-----------|---------|-----------|
| 1 | **V14** | fraud < non-fraud | 0.449 | 7.79 | **3.497** |
| 2 | V12 | fraud < non-fraud | 0.437 | 6.63 | 2.897 |
| 3 | V17 | fraud < non-fraud | 0.308 | 8.91 | 2.746 |
| 4 | V10 | fraud < non-fraud | 0.414 | 5.45 | 2.255 |
| 5 | V3 | fraud < non-fraud | 0.412 | 4.83 | 1.989 |

**Selected donor feature:** V14 (used for PSI drift monitoring in script 06).
**Outputs:** `outputs/feature_discrimination.json`, `outputs/feature_scm_suitability.png`, `outputs/feature_discrimination_ranked.png`, `outputs/feature_discrimination_densities.png`

### 06 — Temporal Validation (`06_temporal_validation.py`)
Walk-forward cross-validation over time to assess real-world model stability. The 48-hour dataset is split into 6 × 8-hour blocks, with 4 expanding-window folds (train on B0, validate on B1, test on B2; then expand).

| PSI range | Signal |
|-----------|--------|
| < 0.10 | Stable |
| 0.10 – 0.20 | Monitor |
| > 0.20 | Retrain |

**Outputs:** `outputs/temporal_validation_summary.json`, `outputs/temporal_fold_metrics.csv`, `outputs/temporal_metrics_by_fold.png`, `outputs/temporal_prec_recall_by_fold.png`, `outputs/temporal_psi_drift.png`, `outputs/temporal_vs_random_comparison.png`

---

## 9. Acknowledgment

Dataset provided by the Machine Learning Group at Université Libre de Bruxelles (ULB), hosted on Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The PCA-transformed features in this dataset originate from a collaboration between Worldline and the ULB Machine Learning Group on big data mining and fraud detection.

Presentation deck: `credit_card_fraud_detection.pptx` — covers the full pipeline, methodology, results, and key takeaways.
