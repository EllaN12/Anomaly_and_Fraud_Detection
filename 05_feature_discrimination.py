"""
Anomaly and Fraud Detection -- 05 FEATURE DISCRIMINATION
============================================================
Evaluates all V1–V28 PCA components objectively to identify the
best donor-stratification variable for the Synthetic Control Method
(script 05). No feature is pre-selected; the analysis determines
the winner using a principled two-factor SCM suitability score.

SCM Suitability Score
---------------------
A good donor-stratification feature must satisfy two conditions:

  1. High discrimination power — the feature separates fraud from
     non-fraud transactions (measured by |AUROC − 0.5|).

  2. Fraud sits far from the non-fraud centre — measured by the
     standardised mean difference (Cohen's d-style):
         std_separation = (fraud_mean − nf_mean) / nf_std
     A large |std_separation| means the fraud cohort occupies an
     extreme region of the non-fraud distribution, so that decile
     groups of non-fraud transactions span a true "fraud-like → typical"
     spectrum rather than all clustering near the centre.

  Combined:  scm_score = disc_power × |std_separation|

Direction filter: only "fraud < non-fraud" features are eligible,
so the fraud cohort sits at the *negative* extreme — decile groups
then naturally run from most-fraud-like (low values) to most-typical
(near zero), giving the SLSQP optimiser a rich, ordered donor pool.

Outputs
-------
  feature_scm_suitability.png     -- all V features ranked by scm_score
  feature_discrimination_ranked.png  -- ranked by raw disc_power only
  feature_discrimination_densities.png -- density grid, top 8 by scm_score
  feature_discrimination.json     -- full scores for all features
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------- #
# PATHS                                                                        #
# --------------------------------------------------------------------------- #
ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _find_csv(name: str = "creditcard.csv") -> Path:
    for candidate in [
        ROOT / "Raw_Data" / name,
        ROOT / name,
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found under {ROOT}")

RAW_CSV  = _find_csv()
V_COLS   = [f"V{i}" for i in range(1, 29)]   # PCA components only

# Palette
NAVY   = "#1E2761"
RED    = "#C0392B"
ORANGE = "#E67E22"
GREEN  = "#1E8449"
LIGHT  = "#EDF0FB"
GRAY   = "#64748B"


# --------------------------------------------------------------------------- #
# 1.  COMPUTE SCORES FOR ALL PCA COMPONENTS                                   #
# --------------------------------------------------------------------------- #

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every V feature compute:
      - auroc           : P(fraud score > non-fraud score)
      - disc_power      : |auroc − 0.5|
      - direction       : which class has higher values
      - std_separation  : (fraud_mean − nf_mean) / nf_std   [signed]
      - scm_score       : disc_power × |std_separation|
                          (used for SCM donor-feature selection)
    """
    fraud    = df[df["Class"] == 1]
    nonfraud = df[df["Class"] == 0]

    rows = []
    for col in V_COLS:
        auroc    = roc_auc_score(df["Class"], df[col])
        power    = abs(auroc - 0.5)
        nf_mean  = nonfraud[col].mean()
        nf_std   = nonfraud[col].std()
        f_mean   = fraud[col].mean()
        std_sep  = (f_mean - nf_mean) / nf_std
        rows.append({
            "feature":        col,
            "auroc":          round(auroc,   4),
            "disc_power":     round(power,   4),
            "direction":      "fraud < non-fraud" if auroc < 0.5 else "fraud > non-fraud",
            "fraud_mean":     round(f_mean,  4),
            "nf_mean":        round(nf_mean, 4),
            "nf_std":         round(nf_std,  4),
            "std_separation": round(std_sep, 4),
            "abs_sep":        round(abs(std_sep), 4),
            "scm_score":      round(power * abs(std_sep), 4),
        })

    return pd.DataFrame(rows).sort_values("scm_score", ascending=False).reset_index(drop=True)


def select_donor_feature(scores: pd.DataFrame) -> str:
    """
    Select the best SCM donor-stratification feature:
      - Must be 'fraud < non-fraud' (fraud cohort at the negative extreme)
      - Rank by scm_score (disc_power × |std_separation|)
      - Return the top-ranked feature name
    """
    eligible = scores[scores["direction"] == "fraud < non-fraud"]
    return eligible.sort_values("scm_score", ascending=False).iloc[0]["feature"]


# --------------------------------------------------------------------------- #
# 2.  PLOT 1 — SCM SUITABILITY RANKING (primary chart)                        #
# --------------------------------------------------------------------------- #

def plot_scm_suitability(scores: pd.DataFrame, winner: str) -> None:
    """
    Horizontal bar chart: all V features ranked by scm_score.
    Winner annotated; ineligible features (fraud > non-fraud) muted.
    """
    eligible = scores[scores["direction"] == "fraud < non-fraud"].sort_values(
        "scm_score", ascending=True
    )
    ineligible = scores[scores["direction"] != "fraud < non-fraud"].sort_values(
        "scm_score", ascending=True
    )
    all_sorted = pd.concat([ineligible, eligible], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 9))

    colors, alphas = [], []
    for _, r in all_sorted.iterrows():
        if r["feature"] == winner:
            colors.append(RED);   alphas.append(1.0)
        elif r["direction"] == "fraud < non-fraud":
            colors.append(NAVY);  alphas.append(0.80)
        else:
            colors.append(GRAY);  alphas.append(0.35)

    bars = ax.barh(
        all_sorted["feature"], all_sorted["scm_score"],
        color=colors, alpha=0.85, edgecolor="white", linewidth=0.5, height=0.65,
    )

    # Annotate winner
    w_row   = all_sorted[all_sorted["feature"] == winner].iloc[0]
    w_score = w_row["scm_score"]
    ax.annotate(
        f"← {winner}  selected\n"
        f"   disc={w_row['disc_power']:.3f}  |sep|={w_row['abs_sep']:.2f}σ",
        xy=(w_score, winner),
        xytext=(w_score + 0.12, winner),
        fontsize=10, color=RED, fontweight="bold", va="center",
        arrowprops=dict(arrowstyle="-", color=RED, lw=0),
    )

    # Divider between eligible and ineligible
    n_inelig = len(ineligible)
    ax.axhline(n_inelig - 0.5, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(
        ax.get_xlim()[1] * 0.02 if ax.get_xlim()[1] > 0 else 0.02,
        n_inelig - 0.5 + 0.3,
        "▲ eligible: fraud < non-fraud direction",
        fontsize=8, color=NAVY, alpha=0.7,
    )
    ax.text(
        ax.get_xlim()[1] * 0.02 if ax.get_xlim()[1] > 0 else 0.02,
        n_inelig - 0.5 - 0.7,
        "▼ ineligible: fraud > non-fraud (muted)",
        fontsize=8, color=GRAY, alpha=0.7,
    )

    ax.set_xlabel(
        "SCM Suitability Score  =  disc_power  ×  |std_separation|\n"
        "(higher = better donor-stratification axis for Synthetic Control)",
        fontsize=11,
    )
    ax.set_title(
        "PCA Component Selection for SCM Donor Stratification\n"
        "Criterion: discrimination power × fraud–nonfraud standardised separation",
        fontsize=13, fontweight="bold",
    )
    ax.set_facecolor(LIGHT)
    fig.patch.set_facecolor("white")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(left=0)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=RED,  label=f"{winner} — selected (highest SCM score)"),
        Patch(facecolor=NAVY, label="eligible: fraud < non-fraud direction"),
        Patch(facecolor=GRAY, alpha=0.4, label="ineligible: fraud > non-fraud"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_scm_suitability.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: feature_scm_suitability.png")


# --------------------------------------------------------------------------- #
# 3.  PLOT 2 — DISCRIMINATION POWER ONLY (secondary, for reference)           #
# --------------------------------------------------------------------------- #

def plot_disc_power(scores: pd.DataFrame, winner: str) -> None:
    by_disc = scores.sort_values("disc_power", ascending=False).reset_index(drop=True)
    features = by_disc["feature"].tolist()
    powers   = by_disc["disc_power"].tolist()
    colors   = [RED if f == winner else NAVY for f in features]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(features)), powers, color=colors, alpha=0.82,
                  edgecolor="white", linewidth=0.6, width=0.7)

    # Annotate winner position on this chart
    w_idx = features.index(winner)
    ax.annotate(
        f"{winner}\n(rank #{w_idx+1} by disc_power\n= {powers[w_idx]:.3f})",
        xy=(w_idx, powers[w_idx]),
        xytext=(w_idx + 2.5, powers[w_idx] + 0.02),
        fontsize=10, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    )

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("|AUROC − 0.5|  (discrimination power)", fontsize=12)
    ax.set_title(
        "Raw Discrimination Power by PCA Component\n"
        f"Selected feature ({winner}) highlighted — ranked by SCM suitability score, not disc_power alone",
        fontsize=12, fontweight="bold",
    )
    ax.axhline(0.10, color=GRAY, linestyle="--", linewidth=1, label="|AUROC−0.5|=0.10 (weak)")
    ax.axhline(0.20, color=GRAY, linestyle=":",  linewidth=1, label="|AUROC−0.5|=0.20 (moderate)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(powers) * 1.22)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_facecolor(LIGHT)
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_discrimination_ranked.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: feature_discrimination_ranked.png")


# --------------------------------------------------------------------------- #
# 4.  PLOT 3 — DENSITY GRID: top 8 by scm_score                              #
# --------------------------------------------------------------------------- #

def plot_density_grid(df: pd.DataFrame, scores: pd.DataFrame, winner: str) -> None:
    top8 = scores.head(8)["feature"].tolist()

    fraud    = df[df["Class"] == 1]
    nonfraud = df[df["Class"] == 0]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Top 8 PCA Components by SCM Suitability Score\n"
        "Fraud vs Non-Fraud Distribution  ·  Selected feature outlined in red",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.35)

    for idx, feat in enumerate(top8):
        ax  = fig.add_subplot(gs[idx // 4, idx % 4])
        row = scores[scores["feature"] == feat].iloc[0]

        lo  = df[feat].quantile(0.01)
        hi  = df[feat].quantile(0.99)
        xs  = np.linspace(lo, hi, 400)

        for subset, label, color, lw in [
            (nonfraud, "Non-Fraud", NAVY, 1.8),
            (fraud,    "Fraud",     RED,  2.2),
        ]:
            vals = subset[feat].clip(lo, hi).values
            if len(np.unique(vals)) > 5:
                kde = gaussian_kde(vals, bw_method=0.3)
                ys  = kde(xs)
                ax.fill_between(xs, ys, alpha=0.18, color=color)
                ax.plot(xs, ys, color=color, linewidth=lw, label=label)

        is_winner = (feat == winner)
        for spine in ax.spines.values():
            spine.set_edgecolor(RED if is_winner else GRAY)
            spine.set_linewidth(2.5 if is_winner else 0.8)

        rank = scores[scores["feature"] == feat].index[0] + 1
        ax.set_title(
            f"{'★ ' if is_winner else ''}#{rank}  {feat}\n"
            f"scm={row['scm_score']:.3f}  |sep|={row['abs_sep']:.2f}σ  "
            f"disc={row['disc_power']:.3f}",
            fontsize=9.5,
            color=RED if is_winner else "black",
            fontweight="bold" if is_winner else "normal",
        )
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_discrimination_densities.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: feature_discrimination_densities.png")


# --------------------------------------------------------------------------- #
# 5.  MAIN                                                                     #
# --------------------------------------------------------------------------- #

def main() -> None:
    print(f"[07] Loading {RAW_CSV.name} ...")
    df = pd.read_csv(RAW_CSV)
    print(f"     {len(df):,} rows  |  {int(df['Class'].sum())} fraud  "
          f"({df['Class'].mean()*100:.3f}%)")

    print("\n[07] Computing SCM suitability scores for all PCA components ...")
    scores = compute_scores(df)

    winner = select_donor_feature(scores)

    print(f"\n  {'Rank':>4}  {'Feature':>8}  {'Direction':>20}  "
          f"{'disc_power':>10}  {'|sep|σ':>7}  {'scm_score':>10}")
    for i, r in scores.head(12).iterrows():
        tag = "  ◄ selected" if r["feature"] == winner else ""
        print(f"  {i+1:>4}  {r['feature']:>8}  {r['direction']:>20}  "
              f"{r['disc_power']:>10.4f}  {r['abs_sep']:>7.4f}  "
              f"{r['scm_score']:>10.4f}{tag}")

    scores.to_json(OUT_DIR / "feature_discrimination.json", orient="records", indent=2)

    print(f"\n[07] Selected feature: {winner}")
    w = scores[scores["feature"] == winner].iloc[0]
    print(f"     disc_power={w['disc_power']:.4f}  "
          f"|std_sep|={w['abs_sep']:.4f}σ  "
          f"direction={w['direction']}")

    print("\n[07] Generating plots ...")
    plot_scm_suitability(scores, winner)
    plot_disc_power(scores, winner)
    plot_density_grid(df, scores, winner)

    print(f"\n[07] Done.  Recommended SCM donor feature: {winner}")


if __name__ == "__main__":
    main()
