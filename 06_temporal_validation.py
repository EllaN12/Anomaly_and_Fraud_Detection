"""
LEARNING LAB 17 (Python port) -- 06 TEMPORAL VALIDATION
=========================================================
Walk-forward temporal validation comparing Isolation Forest and
class-balanced Random Forest against a conventional random split.

PSI is computed across the top-N most discriminating PCA features
(ranked by disc_power from 07_feature_discrimination.json, or inline
if the JSON is not available yet). This gives a fuller picture of
feature-space drift than a single hardcoded feature.

PSI thresholds (industry standard):
  < 0.10  — stable (no action)
  0.10–0.20 — caution (monitor)
  > 0.20  — retrain signal

Fold structure (8-hour blocks over 48 h):
  Fold 1 : Train=B0         | Val=B1 | Test=B2
  Fold 2 : Train=B0+B1      | Val=B2 | Test=B3
  Fold 3 : Train=B0+B1+B2   | Val=B3 | Test=B4
  Fold 4 : Train=B0+B1+B2+B3| Val=B4 | Test=B5
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# 0.  PATHS                                                                   #
# --------------------------------------------------------------------------- #
ROOT         = Path(__file__).resolve().parent
OUT_DIR      = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV      = ROOT / "Raw_Data" / "creditcard.csv"
BLOCK_S      = 8 * 3600          # 8-hour blocks
RANDOM_STATE = 42
MAX_NF_RF    = 15_000            # cap non-fraud rows used to train RF per fold
N_PSI_FEATS  = 5                 # how many top discriminating features to track for PSI

FEAT_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
V_COLS    = [f"V{i}" for i in range(1, 29)]

# --------------------------------------------------------------------------- #
# 1.  UTILITY FUNCTIONS                                                        #
# --------------------------------------------------------------------------- #

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — measures feature distribution shift."""
    bkpts = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(bkpts) < 3:
        return 0.0
    eps = 1e-4
    exp_pct = (np.histogram(expected, bins=bkpts)[0].astype(float) + eps)
    act_pct = (np.histogram(actual,   bins=bkpts)[0].astype(float) + eps)
    exp_pct /= exp_pct.sum()
    act_pct /= act_pct.sum()
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def load_top_psi_features(n: int = N_PSI_FEATS) -> list[str]:
    """
    Return the top-n V features by disc_power (|AUROC − 0.5|).
    Primary source: feature_discrimination.json produced by script 07.
    Fallback: compute disc_power inline from the CSV (slower, used when
    script 07 has not been run yet).
    """
    json_path = OUT_DIR / "feature_discrimination.json"
    if json_path.exists():
        records = json.loads(json_path.read_text())
        # sort descending by disc_power, take top-n
        sorted_r = sorted(records, key=lambda r: r["disc_power"], reverse=True)
        return [r["feature"] for r in sorted_r[:n]]

    # ── inline fallback ──────────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score as _auroc
    df_tmp = pd.read_csv(RAW_CSV, usecols=V_COLS + ["Class"])
    powers = {}
    for col in V_COLS:
        au = _auroc(df_tmp["Class"], df_tmp[col])
        powers[col] = abs(au - 0.5)
    ranked = sorted(powers, key=powers.get, reverse=True)
    return ranked[:n]


def compute_psi_multi(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    features: list[str],
    bins: int = 10,
) -> dict[str, float]:
    """
    Compute PSI for each feature in *features* between train and test blocks.
    Returns {feature: psi_value}.
    """
    return {feat: compute_psi(train_df[feat].values, test_df[feat].values, bins)
            for feat in features}


def f1_optimal_threshold(y_true: np.ndarray, scores: np.ndarray):
    """Threshold that maximises F1 on the given set."""
    from sklearn.metrics import precision_recall_curve
    p, r, thr = precision_recall_curve(y_true, scores)
    p, r = p[:-1], r[:-1]
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1.0), 0.0)
    idx = int(np.argmax(f1))
    return float(thr[idx])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Compute PR AUC, F1, Precision, Recall, MCC at a fixed threshold."""
    preds = (scores >= threshold).astype(int)
    return {
        "pr_auc":    round(float(average_precision_score(y_true, scores)), 4),
        "f1":        round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "mcc":       round(float(matthews_corrcoef(y_true, preds)), 4),
        "threshold": round(threshold, 6),
    }


# --------------------------------------------------------------------------- #
# 2.  MODEL RUNNERS (one fold at a time)                                       #
# --------------------------------------------------------------------------- #

def run_iforest(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """IsoForest: fit on train, calibrate threshold on val, score on test."""
    fraud_rate = float(train["Class"].mean())

    scaler  = StandardScaler().fit(train[FEAT_COLS])
    Xtr     = scaler.transform(train[FEAT_COLS])
    Xval    = scaler.transform(val[FEAT_COLS])
    Xte     = scaler.transform(test[FEAT_COLS])

    clf = IsolationForest(
        n_estimators=100,
        contamination=max(fraud_rate, 1e-4),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(Xtr)

    val_scores  = -clf.score_samples(Xval)
    test_scores = -clf.score_samples(Xte)

    if int(val["Class"].sum()) > 0:
        thr = f1_optimal_threshold(val["Class"].values, val_scores)
    else:
        thr = float(np.percentile(val_scores, 100 * (1 - fraud_rate)))

    return evaluate(test["Class"].values, test_scores, thr)


def run_rf(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """RF balanced: subsample non-fraud for speed; calibrate threshold on val."""
    # Keep all fraud, cap non-fraud to MAX_NF_RF
    fraud_tr   = train[train["Class"] == 1]
    nonfraud_tr = train[train["Class"] == 0]
    if len(nonfraud_tr) > MAX_NF_RF:
        nonfraud_tr = nonfraud_tr.sample(MAX_NF_RF, random_state=RANDOM_STATE)
    train_sub = pd.concat([fraud_tr, nonfraud_tr], ignore_index=True)

    scaler = StandardScaler().fit(train_sub[FEAT_COLS])
    Xtr    = scaler.transform(train_sub[FEAT_COLS])
    ytr    = train_sub["Class"].values
    Xval   = scaler.transform(val[FEAT_COLS])
    Xte    = scaler.transform(test[FEAT_COLS])

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)

    val_proba  = clf.predict_proba(Xval)[:, 1]
    test_proba = clf.predict_proba(Xte)[:, 1]

    if int(val["Class"].sum()) > 0:
        thr = f1_optimal_threshold(val["Class"].values, val_proba)
    else:
        thr = 0.5

    return evaluate(test["Class"].values, test_proba, thr)


# --------------------------------------------------------------------------- #
# 3.  RANDOM SPLIT BASELINE                                                    #
# --------------------------------------------------------------------------- #

def random_split_baseline(df: pd.DataFrame) -> dict:
    """
    Conventional 60/20/20 stratified split.
    Threshold calibrated on validation set (same discipline as temporal folds).
    """
    print("\n[06] Random split baseline (60/20/20 stratified) ...")
    X = df[FEAT_COLS].values
    y = df["Class"].values

    # 60 train / 40 temp
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    # 20 val / 20 test from the 40%
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE
    )

    # ---------- IsoForest ----------
    fraud_rate = float(y_tr.mean())
    scaler_iso = StandardScaler().fit(X_tr)
    Xtr_s      = scaler_iso.transform(X_tr)
    Xval_s     = scaler_iso.transform(X_val)
    Xte_s      = scaler_iso.transform(X_te)

    iso = IsolationForest(
        n_estimators=100, contamination=max(fraud_rate, 1e-4),
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    iso.fit(Xtr_s)
    iso_val_s = -iso.score_samples(Xval_s)
    iso_te_s  = -iso.score_samples(Xte_s)
    iso_thr   = f1_optimal_threshold(y_val, iso_val_s)
    rand_iso  = evaluate(y_te, iso_te_s, iso_thr)
    print(f"   IsoForest  → PR AUC={rand_iso['pr_auc']:.3f}  F1={rand_iso['f1']:.3f}  "
          f"Prec={rand_iso['precision']:.3f}  Rec={rand_iso['recall']:.3f}")

    # ---------- RF balanced ----------
    nf = pd.DataFrame(X_tr[y_tr == 0]).sample(
        min(MAX_NF_RF, (y_tr == 0).sum()), random_state=RANDOM_STATE
    )
    fr_rows = pd.DataFrame(X_tr[y_tr == 1])
    X_tr_sub = np.vstack([fr_rows.values, nf.values])
    y_tr_sub = np.concatenate([np.ones(len(fr_rows)), np.zeros(len(nf))])

    scaler_rf = StandardScaler().fit(X_tr_sub)
    Xtr_r     = scaler_rf.transform(X_tr_sub)
    Xval_r    = scaler_rf.transform(X_val)
    Xte_r     = scaler_rf.transform(X_te)

    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(Xtr_r, y_tr_sub)
    rf_val_p  = rf.predict_proba(Xval_r)[:, 1]
    rf_te_p   = rf.predict_proba(Xte_r)[:, 1]
    rf_thr    = f1_optimal_threshold(y_val, rf_val_p)
    rand_rf   = evaluate(y_te, rf_te_p, rf_thr)
    print(f"   RF balanced→ PR AUC={rand_rf['pr_auc']:.3f}  F1={rand_rf['f1']:.3f}  "
          f"Prec={rand_rf['precision']:.3f}  Rec={rand_rf['recall']:.3f}")

    return {"isoforest": rand_iso, "rf_balanced": rand_rf}


# --------------------------------------------------------------------------- #
# 4.  PLOTS                                                                    #
# --------------------------------------------------------------------------- #

NAVY  = "#1E2761"
RED   = "#C0392B"
GREEN = "#1E8449"
GRAY  = "#64748B"


def plot_fold_metrics(folds_df: pd.DataFrame, rand: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Walk-Forward Temporal Validation — Metrics by Fold\n"
        "(dashed = random-split baseline)",
        fontsize=13, fontweight="bold",
    )
    folds = folds_df["fold"].values

    for ax, metric, ylabel in zip(
        axes,
        ["pr_auc", "f1"],
        ["PR AUC", "F1 Score"],
    ):
        ax.plot(folds, folds_df[f"iso_{metric}"], "o-", color=RED,
                linewidth=2, markersize=8, label="IsoForest (temporal)")
        ax.plot(folds, folds_df[f"rf_{metric}"],  "s-", color=NAVY,
                linewidth=2, markersize=8, label="RF balanced (temporal)")
        ax.axhline(rand["isoforest"][metric],  color=RED,  linestyle="--",
                   alpha=0.55, linewidth=1.5,
                   label=f"IsoForest random ({rand['isoforest'][metric]:.3f})")
        ax.axhline(rand["rf_balanced"][metric], color=NAVY, linestyle="--",
                   alpha=0.55, linewidth=1.5,
                   label=f"RF random ({rand['rf_balanced'][metric]:.3f})")
        ax.set_xlabel("Fold", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=12)
        ax.set_xticks(folds)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "temporal_metrics_by_fold.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_psi_heatmap(folds_df: pd.DataFrame, psi_features: list[str]) -> None:
    """
    Two-panel PSI figure:
      Top   — heatmap: rows=features, cols=folds, cell colour = PSI severity
      Bottom — line plot: max and mean PSI per fold with 0.10 / 0.20 thresholds
    """
    # ── build matrix (features × folds) ────────────────────────────────────
    psi_cols = [f"psi_{f.lower()}" for f in psi_features]
    mat = folds_df[psi_cols].values.T          # shape (n_feat, n_folds)
    folds = folds_df["fold"].values

    fig = plt.figure(figsize=(10, 7))
    gs  = plt.GridSpec(2, 1, height_ratios=[3, 2], hspace=0.45)

    # ── heatmap ─────────────────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0])
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "psi_cmap",
        ["#EDF0FB", "#E67E22", "#C0392B"],   # light → orange → red
        N=256,
    )
    im = ax_heat.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=0.30,
                        interpolation="nearest")
    plt.colorbar(im, ax=ax_heat, label="PSI value", fraction=0.046, pad=0.04)

    ax_heat.set_xticks(range(len(folds)))
    ax_heat.set_xticklabels([f"Fold {f}" for f in folds], fontsize=11)
    ax_heat.set_yticks(range(len(psi_features)))
    ax_heat.set_yticklabels(psi_features, fontsize=11)
    ax_heat.set_title(
        f"Population Stability Index — Top {len(psi_features)} Discriminating Features\n"
        "  < 0.10 stable  ·  0.10–0.20 caution  ·  > 0.20 retrain",
        fontsize=12, fontweight="bold",
    )

    # annotate cells
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            val  = mat[ri, ci]
            text_col = "white" if val > 0.18 else NAVY
            ax_heat.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                         fontsize=10, color=text_col, fontweight="bold")

    # ── line plot ────────────────────────────────────────────────────────────
    ax_line = fig.add_subplot(gs[1])
    psi_max  = folds_df["psi_max"].values
    psi_mean = folds_df["psi_mean"].values

    ax_line.plot(folds, psi_max,  "o-", color=RED,  linewidth=2, markersize=8,
                 label="Max PSI (most stressed feature)")
    ax_line.plot(folds, psi_mean, "s--", color=NAVY, linewidth=2, markersize=7,
                 label="Mean PSI (avg across top features)")
    ax_line.axhline(0.10, color="#E67E22", linestyle=":", linewidth=1.5,
                    label="0.10 caution threshold")
    ax_line.axhline(0.20, color=RED, linestyle=":", linewidth=1.5,
                    label="0.20 retrain threshold")

    ax_line.set_xlabel("Fold (test block)", fontsize=12)
    ax_line.set_ylabel("PSI", fontsize=12)
    ax_line.set_title("Aggregate PSI per Fold", fontsize=11, fontweight="bold")
    ax_line.set_xticks(folds)
    ax_line.legend(fontsize=9, loc="upper right")
    ax_line.grid(True, alpha=0.3)
    ax_line.set_ylim(bottom=0)

    fig.savefig(OUT_DIR / "temporal_psi_drift.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(folds_df: pd.DataFrame, rand: dict) -> None:
    methods = [
        "IsoForest\n(random split)",
        "IsoForest\n(temporal avg)",
        "RF balanced\n(random split)",
        "RF balanced\n(temporal avg)",
    ]
    colors  = ["#E8A0A0", RED, "#8FA8D4", NAVY]
    pr_aucs = [
        rand["isoforest"]["pr_auc"],
        round(folds_df["iso_pr_auc"].mean(), 4),
        rand["rf_balanced"]["pr_auc"],
        round(folds_df["rf_pr_auc"].mean(), 4),
    ]
    f1s = [
        rand["isoforest"]["f1"],
        round(folds_df["iso_f1"].mean(), 4),
        rand["rf_balanced"]["f1"],
        round(folds_df["rf_f1"].mean(), 4),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Random Split vs. Temporal Walk-Forward — Methodology Comparison\n"
        "(lighter bars = random split  ·  darker bars = temporal)",
        fontsize=13, fontweight="bold",
    )
    x = np.arange(len(methods))

    for ax, vals, ylabel in zip(axes, [pr_aucs, f1s], ["PR AUC", "F1 Score"]):
        bars = ax.bar(x, vals, width=0.55, color=colors, edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=12)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "temporal_vs_random_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_breakdown(folds_df: pd.DataFrame, rand: dict) -> None:
    """Side-by-side Precision and Recall across folds for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Precision and Recall Across Temporal Folds",
                 fontsize=13, fontweight="bold")
    folds = folds_df["fold"].values

    for ax, metric, ylabel in zip(axes, ["precision", "recall"], ["Precision", "Recall"]):
        ax.plot(folds, folds_df[f"iso_{metric}"], "o-", color=RED,
                linewidth=2, markersize=8, label="IsoForest")
        ax.plot(folds, folds_df[f"rf_{metric}"],  "s-", color=NAVY,
                linewidth=2, markersize=8, label="RF balanced")
        ax.axhline(rand["isoforest"][metric],  color=RED,  linestyle="--",
                   alpha=0.5, linewidth=1.5)
        ax.axhline(rand["rf_balanced"][metric], color=NAVY, linestyle="--",
                   alpha=0.5, linewidth=1.5)
        ax.set_xlabel("Fold", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=12)
        ax.set_xticks(folds)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "temporal_prec_recall_by_fold.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5.  MAIN                                                                     #
# --------------------------------------------------------------------------- #

def main() -> None:
    print(f"[06] Loading {RAW_CSV.name} ...")
    df = pd.read_csv(RAW_CSV)
    print(f"     shape={df.shape}  fraud={int(df['Class'].sum())}  "
          f"time_range={df['Time'].min():.0f}–{df['Time'].max():.0f}s")

    # ── block summary ──────────────────────────────────────────────────────── #
    max_block = int(df["Time"].max() // BLOCK_S) + 1
    print(f"\n[06] {max_block} blocks of 8 h each:")
    print(f"  {'Blk':>3}  {'Hours':>9}  {'N':>7}  {'Fraud':>6}  {'Rate':>8}")
    for b in range(max_block):
        mask = (df["Time"] >= b * BLOCK_S) & (df["Time"] < (b + 1) * BLOCK_S)
        sub  = df[mask]
        print(f"  {b:>3}  {b*8:>3}–{(b+1)*8:>2}h  "
              f"{len(sub):>7}  {int(sub['Class'].sum()):>6}  "
              f"{sub['Class'].mean()*100:>7.3f}%")

    # ── 4 walk-forward folds ───────────────────────────────────────────────── #
    def get_block(b: int) -> pd.DataFrame:
        mask = (df["Time"] >= b * BLOCK_S) & (df["Time"] < (b + 1) * BLOCK_S)
        return df[mask].copy()

    n_folds = max_block - 2   # 4 folds for 6 blocks
    print(f"\n[06] Walk-forward validation — {n_folds} folds:")

    # Determine which features to track for PSI (done once, outside loop)
    psi_features = load_top_psi_features(N_PSI_FEATS)
    print(f"\n[06] PSI features (top {N_PSI_FEATS} by disc_power): "
          f"{', '.join(psi_features)}")

    fold_records = []
    for k in range(n_folds):
        train_df = pd.concat([get_block(b) for b in range(k + 1)],
                             ignore_index=True)
        val_df   = get_block(k + 1)
        test_df  = get_block(k + 2)

        print(f"\n  ── Fold {k+1} "
              f"(train=B0–B{k}, val=B{k+1}, test=B{k+2}) ──")
        print(f"     train: {len(train_df):>7} rows  "
              f"{int(train_df['Class'].sum())} fraud  "
              f"rate={train_df['Class'].mean()*100:.3f}%")
        print(f"     val  : {len(val_df):>7} rows  "
              f"{int(val_df['Class'].sum())} fraud")
        print(f"     test : {len(test_df):>7} rows  "
              f"{int(test_df['Class'].sum())} fraud")

        iso_m    = run_iforest(train_df, val_df, test_df)
        rf_m     = run_rf(train_df, val_df, test_df)
        psi_dict = compute_psi_multi(train_df, test_df, psi_features)
        psi_max  = max(psi_dict.values())
        psi_mean = sum(psi_dict.values()) / len(psi_dict)

        print(f"     IsoForest   → PR AUC={iso_m['pr_auc']:.3f}  "
              f"F1={iso_m['f1']:.3f}  Prec={iso_m['precision']:.3f}  "
              f"Rec={iso_m['recall']:.3f}")
        print(f"     RF balanced → PR AUC={rf_m['pr_auc']:.3f}  "
              f"F1={rf_m['f1']:.3f}  Prec={rf_m['precision']:.3f}  "
              f"Rec={rf_m['recall']:.3f}")

        psi_parts = "  ".join(f"{f}={v:.3f}" for f, v in psi_dict.items())
        print(f"     PSI         : {psi_parts}")
        print(f"     PSI max={psi_max:.4f}  mean={psi_mean:.4f}"
              f"{'  ⚠ caution' if psi_max > 0.10 else ''}"
              f"{'  ⛔ retrain' if psi_max > 0.20 else ''}")

        per_feat_psi = {f"psi_{f.lower()}": round(v, 4) for f, v in psi_dict.items()}
        fold_records.append({
            "fold":          k + 1,
            "n_train":       len(train_df),
            "n_fraud_train": int(train_df["Class"].sum()),
            "n_fraud_test":  int(test_df["Class"].sum()),
            "psi_max":       round(psi_max,  4),
            "psi_mean":      round(psi_mean, 4),
            **per_feat_psi,
            **{f"iso_{kk}":  v for kk, v in iso_m.items()},
            **{f"rf_{kk}":   v for kk, v in rf_m.items()},
        })

    folds_df = pd.DataFrame(fold_records)
    folds_df.to_csv(OUT_DIR / "temporal_fold_metrics.csv", index=False)

    # ── random split baseline ──────────────────────────────────────────────── #
    rand = random_split_baseline(df)

    # ── aggregate summary ─────────────────────────────────────────────────── #
    summary = {
        "temporal_isoforest": {
            "mean_pr_auc": round(float(folds_df["iso_pr_auc"].mean()), 4),
            "std_pr_auc":  round(float(folds_df["iso_pr_auc"].std()),  4),
            "mean_f1":     round(float(folds_df["iso_f1"].mean()),     4),
            "std_f1":      round(float(folds_df["iso_f1"].std()),      4),
            "mean_recall": round(float(folds_df["iso_recall"].mean()), 4),
            "mean_prec":   round(float(folds_df["iso_precision"].mean()), 4),
        },
        "temporal_rf_balanced": {
            "mean_pr_auc": round(float(folds_df["rf_pr_auc"].mean()), 4),
            "std_pr_auc":  round(float(folds_df["rf_pr_auc"].std()),  4),
            "mean_f1":     round(float(folds_df["rf_f1"].mean()),     4),
            "std_f1":      round(float(folds_df["rf_f1"].std()),      4),
            "mean_recall": round(float(folds_df["rf_recall"].mean()), 4),
            "mean_prec":   round(float(folds_df["rf_precision"].mean()), 4),
        },
        "random_split_isoforest":   rand["isoforest"],
        "random_split_rf_balanced": rand["rf_balanced"],
        "psi_features_tracked": psi_features,
        "psi_summary_by_fold":  folds_df[["fold", "psi_max", "psi_mean"]].to_dict("records"),
        "psi_per_feature_by_fold": [
            {
                "fold": row["fold"],
                **{f: row[f"psi_{f.lower()}"] for f in psi_features},
            }
            for _, row in folds_df.iterrows()
        ],
    }

    print("\n[06] ── Summary ────────────────────────────────────────────")
    print(f"  {'Method':<30} {'PR AUC':>8} {'±':>6} {'F1':>8} {'±':>6}")
    print(f"  {'IsoForest (random split)':<30} "
          f"{rand['isoforest']['pr_auc']:>8.3f} {'':>6} "
          f"{rand['isoforest']['f1']:>8.3f}")
    print(f"  {'IsoForest (temporal avg)':<30} "
          f"{summary['temporal_isoforest']['mean_pr_auc']:>8.3f} "
          f"±{summary['temporal_isoforest']['std_pr_auc']:>5.3f} "
          f"{summary['temporal_isoforest']['mean_f1']:>8.3f} "
          f"±{summary['temporal_isoforest']['std_f1']:>5.3f}")
    print(f"  {'RF balanced (random split)':<30} "
          f"{rand['rf_balanced']['pr_auc']:>8.3f} {'':>6} "
          f"{rand['rf_balanced']['f1']:>8.3f}")
    print(f"  {'RF balanced (temporal avg)':<30} "
          f"{summary['temporal_rf_balanced']['mean_pr_auc']:>8.3f} "
          f"±{summary['temporal_rf_balanced']['std_pr_auc']:>5.3f} "
          f"{summary['temporal_rf_balanced']['mean_f1']:>8.3f} "
          f"±{summary['temporal_rf_balanced']['std_f1']:>5.3f}")

    with open(OUT_DIR / "temporal_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── plots ──────────────────────────────────────────────────────────────── #
    print("\n[06] Generating plots ...")
    plot_fold_metrics(folds_df, rand)
    plot_psi_heatmap(folds_df, psi_features)
    plot_comparison(folds_df, rand)
    plot_precision_recall_breakdown(folds_df, rand)
    print(f"     Saved 4 plots to {OUT_DIR}")

    print("\n[06] Done.")


if __name__ == "__main__":
    main()
