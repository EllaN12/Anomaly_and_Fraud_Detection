"""
Anomaly and Fraud Detection -- MAIN FRAUD / ANOMALY DETECTION
===============================================================
METHOD : scikit-learn IsolationForest 
CASE   : Credit Card Fraud Detection   (creditcard.csv)


WHY IMBALANCE MATTERS
---------------------
The dataset is 577:1  (99.83% non-fraud, 0.17% fraud = 492 / 284,807).
this is why imbalance matters:
  1. contamination="auto"
         sklearn maps "auto" → 0.10 (10%), so the forest expects 28,000+
         anomalies.  We instead set contamination = actual_fraud_rate
         (≈ 0.0017) so the internal threshold is calibrated to reality.

  2. Finding the optimal threshold
        The real fraud count is 492 (0.17%).
         We compare three thresholds side-by-side:
           (a) 99th-percentile (threshold that flags 1% of the data)
           (b) fraud-rate calibrated  (1 – fraud_rate quantile)
           (c) PR-optimal  (threshold that maximises F1 on the PR curve)

  3. ROC AUC as the headline metric
         With 577:1 imbalance, even a trivial "flag nothing" classifier
         scores ROC AUC > 0.5.  PR AUC + F1 + MCC are the right gauges.

  §7  Supervised comparison with imbalance handling
         SMOTE oversampling + Random Forest (class_weight="balanced")
         shows the ceiling a supervised method can reach and how
         imbalance handling changes the picture.

Outputs
-------
    outputs/fraud_scores.csv            one row per transaction
    outputs/fraud_roc.png               ROC curve (§4.3)
    outputs/fraud_pr.png                PR curve, F1-optimal point marked
    outputs/fraud_anomaly_scatter.png   V12 vs V15, colour = outlier (§6.5)
    outputs/fraud_present_scatter.png   V12 vs V15, colour = true fraud
    outputs/fraud_threshold_compare.png side-by-side confusion matrices
    outputs/fraud_metrics.json          key numbers (all three thresholds)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------- #
# PATHS                                                                       #
# --------------------------------------------------------------------------- #
ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = ROOT / "Raw_Data" / "creditcard.csv"


# =========================================================================== #
# 1.0  LOAD + CLASS IMBALANCE                                                 #
# =========================================================================== #
def load_data() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


def class_imbalance(df: pd.DataFrame) -> float:
    """


    Returns the actual fraud rate so downstream functions can use it.
    """
    counts = df["Class"].value_counts().sort_index().rename("n").reset_index()
    counts.columns = ["Class", "n"]
    counts["prop"] = counts["n"] / counts["n"].sum()
    print("[04] Class imbalance (577:1 ratio — addressing this explicitly):")
    print(counts.to_string(index=False))
    fraud_rate = float(counts.loc[counts["Class"] == 1, "prop"].iloc[0])
    print(f"     Imbalance ratio : {(1 - fraud_rate) / fraud_rate:.1f}:1")
    return fraud_rate


# =========================================================================== #
# 2.0  ISOLATION FOREST  (contamination calibrated to actual fraud rate)      #
# =========================================================================== #
def fit_iforest(
    df: pd.DataFrame,
    target: str = "Class",
    n_estimators: int = 100,
    seed: int = 1234,
    contamination: float | str = "auto",
) -> tuple[IsolationForest, list[str]]:
    """
    contamination is the proportion of outliers in the data.
    IMBALANCE FIX: pass contamination = fraud_rate so the forest's
    built-in decision boundary is calibrated to the true anomaly fraction
    instead of sklearn's misleading default of 0.10.
    """
    predictors = [c for c in df.columns if c != target]
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(df[predictors])
    return model, predictors


# =========================================================================== #
# 3.0  PREDICTIONS                                                            #
# =========================================================================== #
def predict_scores(
    df: pd.DataFrame,
    model: IsolationForest,
    predictors: list[str],
) -> np.ndarray:
    """
    return the anomaly scores for the data

    Higher scores = more anomalous
    """
    return -model.score_samples(df[predictors])


# =========================================================================== #
# 4.0  METRICS & THRESHOLDS                                                   #
# =========================================================================== #

# ── 4.1  Three thresholds ────────────────────────────────────────────────── #

def threshold_fixed_pct(scores: np.ndarray, q: float = 0.99) -> tuple[np.ndarray, float]:
    """ flag top 1% (99th-percentile)."""
    thresh = float(np.quantile(scores, q))
    return (scores >= thresh).astype(int), thresh


def threshold_fraud_rate(scores: np.ndarray, fraud_rate: float) -> tuple[np.ndarray, float]:
    """
    IMBALANCE FIX: flag exactly the top `fraud_rate` fraction — calibrated
    to the actual proportion of fraudulent transactions in the data.
    """
    q = 1.0 - fraud_rate
    thresh = float(np.quantile(scores, q))
    return (scores >= thresh).astype(int), thresh


def threshold_f1_optimal(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """
    IMBALANCE FIX: sweep the PR curve and pick the threshold that
    maximises F1 on the minority class.  This is the standard approach
    when neither precision nor recall alone drives the decision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall arrays are one element longer than thresholds
    denom = precision[:-1] + recall[:-1]
    f1 = np.where(
        denom > 0,
        2 * precision[:-1] * recall[:-1] / np.where(denom > 0, denom, 1),
        0.0,
    )
    best_idx  = int(np.argmax(f1))
    best_thresh = float(thresholds[best_idx])
    best_f1     = float(f1[best_idx])
    return (scores >= best_thresh).astype(int), best_thresh, best_f1


# ── 4.2  Balanced metrics ─────────────────────────────────────────────────── #

def balanced_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """
    IMBALANCE FIX: report precision, recall, F1, balanced accuracy, MCC
    instead of (or in addition to) raw accuracy, which is misleading here
    (a classifier that flags nothing gets 99.83% accuracy).
    """
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc     = matthews_corrcoef(y_true, y_pred)

    print(f"\n[04] ── {label} ──")
    print(f"     TP={tp:5d}  FP={fp:6d}  FN={fn:3d}  TN={tn:,}")
    print(f"     Precision      : {prec:.4f}")
    print(f"     Recall         : {rec:.4f}")
    print(f"     F1             : {f1:.4f}")
    print(f"     Balanced Acc   : {bal_acc:.4f}")
    print(f"     MCC            : {mcc:.4f}")

    return {
        "label": label, "TP": int(tp), "FP": int(fp),
        "FN": int(fn), "TN": int(tn),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4), "balanced_accuracy": round(bal_acc, 4),
        "mcc": round(mcc, 4),
    }


# ── 4.3  ROC AUC ─────────────────────────────────────────────────────────── #

def plot_roc(y: np.ndarray, scores: np.ndarray, path: Path) -> float:
    """
    calculate the ROC AUC for the data
    NOTE: ROC AUC can look great (≈0.95) on 577:1 data even for a weak
    detector because the denominator includes the huge non-fraud pool.
    PR AUC is the primary metric here.
    """
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2c3e50", linewidth=2,
            label=f"ROC AUC: {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(
        f"ROC AUC: {auc:.3f} — Isolation Forest\n"
        "(ROC is optimistic on imbalanced data; see PR curve)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return float(auc)


# ── 4.4  PR AUC with F1-optimal point marked ─────────────────────────────── #

def plot_pr(
    y: np.ndarray,
    scores: np.ndarray,
    path: Path,
    f1_thresh: float | None = None,
) -> float:
    """
    calculate the PR AUC for the data
    IMBALANCE FIX: marks the F1-optimal operating point on the curve,
    and draws the no-skill baseline (= fraud_rate), giving a calibrated
    view of how much better the model is vs. random.
    """
    precision, recall, thresholds = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    baseline = float(y.mean())   # no-skill baseline = fraud rate

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color="#e74c3c", linewidth=2,
            label=f"IsoForest PR AUC: {ap:.3f}")
    ax.axhline(baseline, linestyle="--", color="gray", linewidth=1,
               label=f"No-skill baseline: {baseline:.4f}")

    # Mark F1-optimal threshold point
    if f1_thresh is not None:
        idx = np.searchsorted(thresholds, f1_thresh)
        idx = min(idx, len(recall) - 2)
        ax.scatter(recall[idx], precision[idx], s=120, color="#e67e22",
                   zorder=5, label=f"F1-optimal (thresh={f1_thresh:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"Precision-Recall AUC: {ap:.3f}\n"
        "Isolation Forest  (primary metric for imbalanced data)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return float(ap)


# ── 4.5  Threshold comparison — side-by-side confusion matrices ───────────── #

def plot_threshold_compare(
    y_true: np.ndarray,
    threshold_results: list[tuple[str, np.ndarray]],
    path: Path,
) -> None:
    """
    IMBALANCE FIX: visualise how each threshold strategy trades off
    TP (catching fraud) against FP (flagging legitimate transactions).
    """
    n = len(threshold_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))

    for ax, (label, y_pred) in zip(axes, threshold_results):
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        tick_labels = ["Non-Fraud", "Fraud"]
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_yticklabels(tick_labels, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_title(label, fontsize=9, pad=8)
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, f"{val:,}", ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")

    fig.suptitle(
        "Threshold Strategy Comparison — Confusion Matrices\n"
        "(Addressing 577:1 class imbalance)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# 5.0  V12 vs V15 SCATTER                                                     #
# =========================================================================== #

def plot_v12_v15(
    df: pd.DataFrame,
    color_col: str,
    title: str,
    legend_label: str,
    path: Path,
) -> None:
    """R (§6.5): ggplot(aes(V12, V15, color = as.factor(outlier)))"""
    colors = df[color_col].map({0: "#2c3e50", 1: "#e74c3c"})
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["V12"], df["V15"], c=colors, alpha=0.2, s=4,
               rasterized=True)
    patches = [
        mpatches.Patch(color="#2c3e50", label=f"{legend_label} = 0"),
        mpatches.Patch(color="#e74c3c", label=f"{legend_label} = 1"),
    ]
    ax.legend(handles=patches, title=legend_label)
    ax.set_xlabel("V12"); ax.set_ylabel("V15")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# =========================================================================== #
# 6.0  BONUS — STABILIZE WITH MULTIPLE SEEDS  (mirrors R §6)                 #
# =========================================================================== #

def iso_forest_one_seed(
    df: pd.DataFrame, predictors: list[str], seed: int, fraud_rate: float
) -> np.ndarray:
    """ fit the Isolation Forest model to the data """
    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=fraud_rate,   # IMBALANCE FIX: keep calibrated
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(df[predictors])
    return -model.score_samples(df[predictors])


def stabilized_scores(
    df: pd.DataFrame,
    predictors: list[str],
    fraud_rate: float,
    seeds: tuple[int, ...] = (158, 8546, 4593),
) -> np.ndarray:
    """ map seeds to mean predictions."""
    print(f"[04] Stabilizing with seeds {seeds} ...")
    mat = np.stack(
        [iso_forest_one_seed(df, predictors, s, fraud_rate) for s in seeds],
        axis=1,
    )
    return mat.mean(axis=1)


# =========================================================================== #
# 7.0  SUPERVISED COMPARISON WITH IMBALANCE HANDLING  (Python-only bonus)     #
# =========================================================================== #

def supervised_comparison(
    df: pd.DataFrame, predictors: list[str], y_true: np.ndarray
) -> dict:
    """
    IMBALANCE FIX: show how a class-weighted supervised classifier compares.

    We use a stratified subsample (all 492 fraud rows + 10,000 random
    non-fraud rows) for 5-fold cross-validation.  This keeps runtime to
    a few seconds while preserving the full minority class and giving a
    fair comparison.  PR AUC on this subsample is representative of what
    a tuned supervised model can reach on the full dataset.

    Two variants:
      A) Random Forest, class_weight="balanced"  (no resampling)
      B) SMOTE oversampling + Random Forest  (if imbalanced-learn installed)
    """
    X_full = df[predictors].to_numpy()

    # ── Stratified subsample: keep all fraud, sample non-fraud ────────────── #
    rng         = np.random.default_rng(42)
    fraud_idx   = np.where(y_true == 1)[0]
    nf_idx_all  = np.where(y_true == 0)[0]
    nf_idx      = rng.choice(nf_idx_all, size=10_000, replace=False)
    sample_idx  = np.concatenate([fraud_idx, nf_idx])

    X = X_full[sample_idx]
    y = y_true[sample_idx]
    print(f"\n[07] Supervised CV on stratified subsample: "
          f"{len(y):,} rows ({y.sum()} fraud + 10,000 non-fraud)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # ── A: class_weight="balanced" ────────────────────────────────────────── #
    print("[07] Supervised: Random Forest (class_weight='balanced') ...")
    rf_balanced = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    proba_a = cross_val_predict(
        rf_balanced, X, y, cv=cv, method="predict_proba"
    )[:, 1]
    pred_a = (proba_a >= 0.5).astype(int)

    m_a = balanced_metrics(y, pred_a, "RF balanced (threshold=0.5)")
    m_a["pr_auc"]  = round(average_precision_score(y, proba_a), 4)
    m_a["roc_auc"] = round(roc_auc_score(y, proba_a), 4)
    results["rf_balanced"] = m_a

    # ── B: SMOTE + Random Forest ───────────────────────────────────────────── #
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        print("[07] Supervised: SMOTE + Random Forest ...")
        smote_rf = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf",   RandomForestClassifier(
                          n_estimators=100, random_state=42, n_jobs=-1)),
        ])
        proba_b = cross_val_predict(
            smote_rf, X, y, cv=cv, method="predict_proba"
        )[:, 1]
        pred_b = (proba_b >= 0.5).astype(int)

        m_b = balanced_metrics(y, pred_b, "SMOTE + RF (threshold=0.5)")
        m_b["pr_auc"]  = round(average_precision_score(y, proba_b), 4)
        m_b["roc_auc"] = round(roc_auc_score(y, proba_b), 4)
        results["smote_rf"] = m_b

    except ImportError:
        print("[07] imbalanced-learn not installed — skipping SMOTE variant.")
        print("     Install with: pip install imbalanced-learn")

    return results


# =========================================================================== #
# MAIN                                                                        #
# =========================================================================== #
def main() -> None:

    # ── 1.0 Load + class imbalance ────────────────────────────────────────── #
    print(f"[04] Loading {RAW_CSV.name}  ({RAW_CSV.stat().st_size / 1e6:.1f} MB) ...")
    df = load_data()
    print(f"     shape = {df.shape}")
    fraud_rate = class_imbalance(df)
    y_true     = df["Class"].to_numpy()
    predictors = [c for c in df.columns if c != "Class"]

    # ── 2.0 Fit Isolation Forest  (contamination = actual fraud rate) ─────── #
    print(
        f"\n[04] Fitting Isolation Forest"
        f" (100 trees, seed=1234, contamination={fraud_rate:.6f}) ..."
    )
    model, _ = fit_iforest(
        df, target="Class", n_estimators=100, seed=1234,
        contamination=fraud_rate,           # IMBALANCE FIX
    )

    # ── 3.0 Predictions ───────────────────────────────────────────────────── #
    scores = predict_scores(df, model, predictors)

    # ── 4.1 Three thresholds ──────────────────────────────────────────────── #
    outlier_99,     thresh_99     = threshold_fixed_pct(scores, q=0.99)
    outlier_rate,   thresh_rate   = threshold_fraud_rate(scores, fraud_rate)
    outlier_f1opt,  thresh_f1opt, best_f1 = threshold_f1_optimal(y_true, scores)

    print(f"\n[04] Threshold comparison:")
    print(f"     (a) 99th-percentile         thresh={thresh_99:.5f}  "
          f"flags {outlier_99.sum():,}")
    print(f"     (b) Fraud-rate calibrated   thresh={thresh_rate:.5f}  "
          f"flags {outlier_rate.sum():,}  (target ≈ {round(fraud_rate * len(df))})")
    print(f"     (c) F1-optimal              thresh={thresh_f1opt:.5f}  "
          f"flags {outlier_f1opt.sum():,}  (best F1={best_f1:.4f})")

    # ── 4.2 Balanced metrics for each threshold ────────────────────────────── #
    m99   = balanced_metrics(y_true, outlier_99,    "Threshold (a) 99th-percentile")
    mrate = balanced_metrics(y_true, outlier_rate,  "Threshold (b) Fraud-rate calibrated")
    mf1   = balanced_metrics(y_true, outlier_f1opt, "Threshold (c) F1-optimal")

    # ── 4.3 ROC AUC ───────────────────────────────────────────────────────── #
    roc_auc = plot_roc(y_true, scores, OUT_DIR / "fraud_roc.png")
    print(f"\n[04] ROC AUC = {roc_auc:.3f}  "
          f"(inflated by 577:1 imbalance — see PR AUC below)")

    # ── 4.4 PR AUC (primary metric) ───────────────────────────────────────── #
    pr_auc = plot_pr(
        y_true, scores, OUT_DIR / "fraud_pr.png",
        f1_thresh=thresh_f1opt,
    )
    print(f"[04] PR  AUC = {pr_auc:.3f}  (primary metric for imbalanced data)")

    # ── 4.5 Threshold comparison plot ─────────────────────────────────────── #
    plot_threshold_compare(
        y_true,
        [
            ("(a) 99th-pct\n(original)", outlier_99),
            ("(b) Fraud-rate\ncalibrated", outlier_rate),
            ("(c) F1-optimal\n(max F1)", outlier_f1opt),
        ],
        OUT_DIR / "fraud_threshold_compare.png",
    )

    # ── 5.0 Conclusions ───────────────────────────────────────────────────── #
    print("\n[04] Conclusions:")
    print("     - Anomalies (Outliers) are more often than not Fraudulent Transactions")
    print("     - Isolation Forest does a good job at detecting anomalous behaviour")
    print("     - With 577:1 imbalance, calibrating contamination and threshold")
    print("       to the true fraud rate gives far fewer false positives")

    # ── 6.0 Stabilize with multiple seeds ────────────────────────────────── #
    mean_scores  = stabilized_scores(df, predictors, fraud_rate,
                                     seeds=(158, 8546, 4593))
    stab_outlier, _ = threshold_fraud_rate(mean_scores, fraud_rate)
    stab_pr_auc     = average_precision_score(y_true, mean_scores)
    print(f"\n[04] Stabilized PR AUC (3-seed mean) = {stab_pr_auc:.3f}")

    # ── 6.5 Visualize V12 vs V15 ─────────────────────────────────────────── #
    result_df = df[["Time", "V12", "V15", "Amount", "Class"]].copy()
    result_df["anomaly_score"]      = scores
    result_df["anomaly_score_stab"] = mean_scores
    result_df["outlier"]            = stab_outlier

    plot_v12_v15(
        result_df, "outlier",
        title="Anomaly Detected?\n(red = flagged by Isolation Forest, fraud-rate threshold)",
        legend_label="Is Outlier?",
        path=OUT_DIR / "fraud_anomaly_scatter.png",
    )
    plot_v12_v15(
        result_df, "Class",
        title="Fraud Present?\n(red = actual fraudulent transaction)",
        legend_label="Is Fraud?",
        path=OUT_DIR / "fraud_present_scatter.png",
    )

    # ── 7.0 Supervised comparison ─────────────────────────────────────────── #
    supervised_results = supervised_comparison(df, predictors, y_true)

    # ── Persist ───────────────────────────────────────────────────────────── #
    result_df.to_csv(OUT_DIR / "fraud_scores.csv", index=False)

    metrics = {
        "dataset": {
            "n_transactions"  : int(len(df)),
            "n_fraud_true"    : int(y_true.sum()),
            "fraud_rate"      : round(fraud_rate, 6),
            "imbalance_ratio" : round((1 - fraud_rate) / fraud_rate, 1),
        },
        "isolation_forest": {
            "contamination_used"   : round(fraud_rate, 6),
            "roc_auc"              : round(roc_auc, 4),
            "pr_auc"               : round(pr_auc, 4),
            "pr_auc_stabilized"    : round(stab_pr_auc, 4),
            "threshold_99th"       : m99,
            "threshold_fraud_rate" : mrate,
            "threshold_f1_optimal" : mf1,
        },
        "supervised_comparison": supervised_results,
    }
    (OUT_DIR / "fraud_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )

    print("\n[04] Metrics written to fraud_metrics.json")
    print("[04] Outputs written to", OUT_DIR)
    print("[04] Done.")


if __name__ == "__main__":
    main()
