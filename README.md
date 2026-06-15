# Credit Card Fraud Detection — Anomaly Detection Portfolio Project

End-to-end anomaly and fraud detection pipeline on 284,807 real credit card transactions. Seven Python scripts (00–06) cover data preparation, three controlled anomaly experiments, fraud detection with Isolation Forest, objective feature ranking, and temporal walk-forward validation.

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**File:** `Raw_Data/creditcard.csv` (150.8 MB) — **not in git** (exceeds GitHub's 100 MB limit). Download from Kaggle and place in `Raw_Data/` before running the scripts. See [Raw_Data/README.md](Raw_Data/README.md).

| Attribute | Value |
|-----------|-------|
| Transactions | 284,807 |
| Features | 30 PCA components (V1–V28) + Time + Amount |
| Fraud cases | 492 (0.173%) |
| Imbalance ratio | 577 : 1 |

All V1–V28 features are anonymised PCA projections. `Time` is seconds from the first transaction. `Class` is the binary label (1 = fraud).

---

## Project Structure

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

## Setup

```bash
pip install -r requirements.txt
# Optional — enables SMOTE oversampling variant
pip install imbalanced-learn
```

### Run everything

```bash
python3 run_all.py
```

Or run scripts individually in order (00 → 06). All outputs are written to `outputs/`.

---

## Scripts

### 00 — Data Preparation (`00_data_prep.py`)

Loads `creditcard.csv`, reports class imbalance, and generates the transaction amount density plot split by class.

**Key output:** `outputs/amount_by_class_density.png`

| Stat | Value |
|------|-------|
| Rows | 284,807 |
| Fraud rate | 0.173% |
| Imbalance | 577 : 1 |

---

### 01 — Point Anomaly Experiment (`01_experiment_1_point_anomalies.py`)

Synthetic 2-D Gaussian cluster with a single injected outlier. Isolation Forest (99th-percentile threshold) correctly isolates the lone deviant.

**Output:** `outputs/exp1_point_anomalies.{csv,png}`

---

### 02 — Time-Series Anomaly Experiment (`02_experiment_2_timeseries_anomalies.py`)

Linear trend with two injected spikes. Two detection methods compared:

- **Isolation Forest** (IQR-based 70th-pct threshold) → 9 flags
- **STL decomposition + IQR** → 3 flags (catches true spikes precisely)

**Outputs:** `outputs/exp2_timeseries_anomalies.csv`, `outputs/exp2_iforest.png`, `outputs/exp2_stl_decomposition.png`

---

### 03 — Cluster Anomaly Experiment (`03_experiment_3_cluster_anomalies.py`)

Base cluster at (50, 50) + satellite cluster at (80, 80). Isolation Forest with an 80th-pct threshold flags 7 of 35 points — correctly identifying the remote satellite group as collective anomalies.

**Output:** `outputs/exp3_cluster_anomalies.{csv,png}`

---

### 04 — Fraud Anomaly Detection (`04_fraud_anomaly_detection.py`)

Core script. Fits Isolation Forest on the full real dataset and compares unsupervised detection against a supervised Random Forest baseline.

#### Key fix — `contamination` parameter

sklearn's default `contamination='auto'` maps to 0.10 (10%), expecting ~28,480 anomalies. Actual fraud is 492 (0.173%). Setting `contamination = fraud_rate = 0.001727` calibrates the model to the true anomaly fraction.

#### Threshold comparison

| Threshold | Flags | Precision | Recall | F1 |
|-----------|-------|-----------|--------|----|
| 99th-pct | 2,849 | 0.109 | 0.632 | 0.186 |
| Fraud-rate calibrated | 492 | 0.295 | 0.295 | 0.295 |
| F1-optimal | 555 | 0.294 | 0.331 | **0.311** |

#### AUC metrics

| Metric | Value | Note |
|--------|-------|------|
| ROC AUC | 0.955 | Inflated by 577:1 majority dominance |
| PR AUC | 0.189 | Primary metric for imbalanced data |
| PR AUC (3-seed stabilised) | 0.169 | Averaged over seeds 158, 8546, 4593 |

#### Supervised comparison (stratified subsample: 10,492 rows)

| Method | F1 | MCC | Recall | PR AUC |
|--------|----|-----|--------|--------|
| Isolation Forest | 0.311 | 0.311 | 0.331 | 0.189 |
| RF (class_weight='balanced') | **0.905** | **0.903** | **0.854** | **0.916** |

**Key insight:** Labelled training data enables a 3× F1 improvement over unsupervised detection. Use Isolation Forest where labels are unavailable; use supervised RF when they exist.

**Outputs:** `outputs/fraud_metrics.json`, `outputs/fraud_roc.png`, `outputs/fraud_pr.png`, `outputs/fraud_threshold_compare.png`, `outputs/fraud_anomaly_scatter.png`, `outputs/fraud_scores.csv`

---

### 05 — Feature Discrimination (`05_feature_discrimination.py`)

Ranks all 28 PCA components using the **SCM suitability score** — an objective metric that combines discriminatory power and distributional separation, enabling data-driven donor-feature selection without hardcoding.

```
scm_score = disc_power × |std_separation|
disc_power = |AUROC − 0.5|
std_separation = (fraud_mean − nf_mean) / nf_std
```

#### Top 5 ranked features

| Rank | Feature | Direction | disc_power | \|sep\| σ | scm_score |
|------|---------|-----------|-----------|---------|-----------|
| 1 | **V14** | fraud < non-fraud | 0.449 | 7.79 | **3.497** |
| 2 | V12 | fraud < non-fraud | 0.437 | 6.63 | 2.897 |
| 3 | V17 | fraud < non-fraud | 0.308 | 8.91 | 2.746 |
| 4 | V10 | fraud < non-fraud | 0.414 | 5.45 | 2.255 |
| 5 | V3 | fraud < non-fraud | 0.412 | 4.83 | 1.989 |

**Selected donor feature:** V14 (used for PSI drift monitoring in script 06).

**Outputs:** `outputs/feature_discrimination.json`, `outputs/feature_scm_suitability.png`, `outputs/feature_discrimination_ranked.png`, `outputs/feature_discrimination_densities.png`

---

### 06 — Temporal Validation (`06_temporal_validation.py`)

Walk-forward cross-validation over time to assess real-world model stability. The 48-hour dataset is split into 6 × 8-hour blocks, with 4 expanding-window folds.

#### Fold structure

| Fold | Train | Val | Test |
|------|-------|-----|------|
| 1 | B0 | B1 | B2 |
| 2 | B0–B1 | B2 | B3 |
| 3 | B0–B2 | B3 | B4 |
| 4 | B0–B3 | B4 | B5 |

#### Results summary

| Method | PR AUC (avg ± std) | F1 (avg ± std) |
|--------|--------------------|----------------|
| Isolation Forest — temporal | 0.176 ± 0.273 | 0.199 ± 0.277 |
| Isolation Forest — random split | 0.131 | 0.197 |
| RF balanced — temporal | **0.783 ± 0.034** | **0.810 ± 0.026** |
| RF balanced — random split | 0.809 | 0.827 |

**Isolation Forest is temporally unstable** — PR AUC swings from 0.018 to 0.585 across folds. Fraud-rate shifts between 8-hour blocks break the contamination calibration.

**RF balanced is robust** — PR AUC stays in the 0.743–0.818 band with low variance.

#### PSI drift monitoring

Population Stability Index tracked across top-5 discriminating features (V14, V4, V12, V11, V10):

| PSI range | Signal |
|-----------|--------|
| < 0.10 | Stable |
| 0.10 – 0.20 | Monitor |
| > 0.20 | Retrain |

Fold 1 shows significant V12 drift (PSI = 2.93), triggering a retrain signal. Drift stabilises in later folds as the training window grows.

**Outputs:** `outputs/temporal_validation_summary.json`, `outputs/temporal_fold_metrics.csv`, `outputs/temporal_metrics_by_fold.png`, `outputs/temporal_prec_recall_by_fold.png`, `outputs/temporal_psi_drift.png`, `outputs/temporal_vs_random_comparison.png`

---

## Key Findings

1. **Imbalance is the primary challenge.** 577:1 ratio invalidates accuracy and ROC AUC as metrics. Calibrating `contamination` to the true fraud rate and optimising the detection threshold by F1 are the critical fixes for unsupervised detection.

2. **Unsupervised detects; supervised catches.** Isolation Forest achieves F1=0.311 and PR AUC=0.189 with no labels. RF with `class_weight='balanced'` reaches F1=0.905 — a 3× improvement when labelled data is available.

3. **Feature selection must be objective, not assumed.** All 28 PCA components are scored by SCM suitability. V14 wins with score 3.497. Hardcoding a feature without this ranking risks building on a sub-optimal discrimination axis.

4. **Temporal validation exposes model fragility.** Walk-forward CV reveals IsoForest PR AUC variance of ±0.273 — effectively unreliable across time. RF balanced holds ±0.034. PSI drift monitoring across top-5 features provides an early retrain signal.

---

## Technologies

| Library | Purpose |
|---------|---------|
| pandas, numpy | Data loading and manipulation |
| scikit-learn | Isolation Forest, Random Forest, metrics |
| statsmodels | STL time-series decomposition |
| matplotlib | All visualisations |
| imbalanced-learn *(optional)* | SMOTE oversampling |

---

## Presentation

`credit_card_fraud_detection.pptx` — 12-slide deck covering the full pipeline, methodology, results, and key takeaways.
