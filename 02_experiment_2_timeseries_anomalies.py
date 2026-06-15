"""
Anomaly and Fraud Detection -- EXPERIMENT 2
=============================================
METHOD : IsolationForest (point view) + STL decomposition (time view)
CASE   : A near-linear trend with two artificially injected spikes.
"""
#%%
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

HERE = Path(__file__).resolve().parent  # repo root
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 2.0 DATA -- trend with two injected point anomalies                         #
# --------------------------------------------------------------------------- #
def make_data(seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.linspace(20, 40, 30)
    y = x + rng.normal(50, 3, size=30)
    
    y[9] += 40
    y[19] += 40
    dates = pd.date_range("2019-01-01", periods=30, freq="D")
    return pd.DataFrame({"date": dates, "x": x, "y": y})


# --------------------------------------------------------------------------- #
# 3.0 ISOLATION FOREST VIEW                                                   #
# --------------------------------------------------------------------------- #
def iforest_view(df: pd.DataFrame, seed: int = 12345) -> pd.DataFrame:
    model = IsolationForest(
        n_estimators=200, max_samples="auto", random_state=seed
    )
    feats = df[["x", "y"]]
    model.fit(feats)
    anomaly_score = -model.score_samples(feats)

    out = df.copy()
    out["iforest_score"] = anomaly_score
    thresh = np.quantile(anomaly_score, 0.70)  # R used 0.70 here
    out["iforest_outlier"] = (anomaly_score >= thresh).astype(int)
    return out


# --------------------------------------------------------------------------- #
# 6.0 STL / ANOMALIZE-STYLE VIEW                                              #
# --------------------------------------------------------------------------- #
def stl_view(df: pd.DataFrame, iqr_mult: float = 3.0) -> pd.DataFrame:
    """Decompose y into trend+seasonal+remainder, then flag remainder
    points outside [Q1 - k*IQR, Q3 + k*IQR] -- this is the IQR method."""
    y = df["y"].to_numpy()
    # Small series -- period=7 for weekly-ish seasonality, robust STL
    stl = STL(y, period=7, robust=True).fit()
    trend = stl.trend
    seasonal = stl.seasonal
    remainder = stl.resid

    q1, q3 = np.quantile(remainder, [0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
    anomaly = ((remainder < lo) | (remainder > hi)).astype(int)

    out = df.copy()
    out["trend"] = trend
    out["seasonal"] = seasonal
    out["remainder"] = remainder
    out["stl_outlier"] = anomaly
    return out


# --------------------------------------------------------------------------- #
# VIZ                                                                         #
# --------------------------------------------------------------------------- #
def plot_scatter(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = df["iforest_outlier"].map({0: "#2c3e50", 1: "#e74c3c"})
    ax.plot(df["x"], df["y"], color="#95a5a6", linewidth=1, zorder=1)
    ax.scatter(df["x"], df["y"], c=colors, s=60, zorder=2, edgecolor="white")
    ax.set_title("Experiment 2a - Isolation Forest view\n(top 30% anomaly score flagged)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_stl(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(df["date"], df["y"], marker="o")
    axes[0].set_title("Observed")
    axes[1].plot(df["date"], df["trend"], color="#27ae60")
    axes[1].set_title("Trend")
    axes[2].plot(df["date"], df["seasonal"], color="#2980b9")
    axes[2].set_title("Seasonal")
    axes[3].plot(df["date"], df["remainder"], color="#7f8c8d", marker="o")
    outlier_mask = df["stl_outlier"] == 1
    axes[3].scatter(
        df.loc[outlier_mask, "date"],
        df.loc[outlier_mask, "remainder"],
        color="#e74c3c",
        s=70,
        zorder=3,
        label="STL anomaly",
    )
    axes[3].axhline(0, color="black", linewidth=0.5)
    axes[3].set_title("Remainder (red = flagged)")
    axes[3].legend(loc="upper right")
    fig.suptitle("Experiment 2b - STL decomposition (anomalize-style)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    df = make_data()
    df = iforest_view(df)
    df = stl_view(df)

    print("[02] rows flagged by iforest :", int(df["iforest_outlier"].sum()))
    print("[02] rows flagged by STL/IQR :", int(df["stl_outlier"].sum()))

    out_csv = OUT_DIR / "exp2_timeseries_anomalies.csv"
    df.to_csv(out_csv, index=False)

    plot_scatter(df, OUT_DIR / "exp2_iforest.png")
    plot_stl(df, OUT_DIR / "exp2_stl_decomposition.png")
    print(f"[02] Wrote {out_csv.name} + two PNGs")


if __name__ == "__main__":
    main()
