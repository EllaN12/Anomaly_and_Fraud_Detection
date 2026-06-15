"""
-- Anomaly and Fraud Detection -- 00 DATA PREP
=============================================
Loads creditcard.csv, inspects class imbalance, and plots the
Amount-by-Class density distribution.

"""
#%%
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1.0 PATHS                                                                   #
# --------------------------------------------------------------------------- #
HERE    = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# creditcard.csv lives one level above the python/ folder
RAW_CSV = PROJECT / "creditcard.csv"

#%%
# --------------------------------------------------------------------------- #
# 2.0 LOAD                                                                    #
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    """Read creditcard.csv from the project root"""
    df = pd.read_csv(RAW_CSV)
    return df

print(load_data())
#%%
# --------------------------------------------------------------------------- #
# 3.0 CLASS IMBALANCE                                                         #
# --------------------------------------------------------------------------- #
def class_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of fraud and non-fraud cases and calculate the proportion of fraud cases.
    """
    counts = df["Class"].value_counts().sort_index().rename("n").reset_index()
    counts.columns = ["Class", "n"]
    counts["prop"] = counts["n"] / counts["n"].sum()
    return counts


# --------------------------------------------------------------------------- #
# 4.0 AMOUNT vs FRAUD DENSITY PLOT                                            #
# --------------------------------------------------------------------------- #
def plot_amount_by_class(df: pd.DataFrame, path: Path) -> None:
    """
    Plot the amount spent by class density.
    """
    classes   = sorted(df["Class"].unique())
    fig, axes = plt.subplots(len(classes), 1, figsize=(8, 5 * len(classes)),
                             sharex=True)

    colors = {0: "#2c3e50", 1: "#e74c3c"}
    labels = {0: "Non-Fraud (0)", 1: "Fraud (1)"}

    for ax, cls in zip(axes, classes):
        subset = df.loc[df["Class"] == cls, "Amount"]
        # log10 transform (drop zeros to avoid -inf)
        log_amt = np.log10(subset.clip(lower=0.01))
        ax.hist(log_amt, bins=60, density=True, alpha=0.5,
                color=colors[cls], edgecolor="none")
        # kernel-density line
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(log_amt)
        xs  = np.linspace(log_amt.min(), log_amt.max(), 300)
        ax.plot(xs, kde(xs), color=colors[cls], linewidth=2)
        ax.set_title(f"Class {cls} — {labels[cls]}", fontsize=11)
        ax.set_ylabel("Density")

    axes[-1].set_xlabel("Amount (log₁₀ scale, $)")
    # x-tick labels back to dollar values
    xticks = np.arange(-2, 4)
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([f"${10**float(v):,.2f}" for v in xticks])

    fig.suptitle("Fraud by Amount Spent", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5.0 MAIN                                                                    #
# --------------------------------------------------------------------------- #
def main() -> None:
    print(f"[00] Reading {RAW_CSV.name} ...")
    df = load_data()
    print(f"     shape = {df.shape}")
    print(f"     columns = {list(df.columns)}")

    # 1.1 Class imbalance
    imbalance = class_imbalance(df)
    print("\n[00] Class imbalance :")
    print(imbalance.to_string(index=False))

    # 1.2 Amount density by class
    plot_path = OUT_DIR / "amount_by_class_density.png"
    plot_amount_by_class(df, plot_path)
    print(f"\n[00] Saved density plot → {plot_path.name}")

    # Quick numeric summary
    print("\n[00] Numeric summary (Time, Amount, Class):")
    print(df[["Time", "Amount", "Class"]].describe().to_string())

    print("\n[00] Done.")


if __name__ == "__main__":
    main()

# %%
