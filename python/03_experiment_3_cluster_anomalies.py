"""
LEARNING LAB 17 (Python port) -- EXPERIMENT 3
=============================================
METHOD : IsolationForest
CASE   : One dense base cluster + a small satellite cluster.  The
         satellite cluster should be flagged as the collective anomaly.

Mirror of `lab_17_experiment_3_cluster_anomalies.R`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_data(seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.DataFrame(
        {"x": rng.normal(50, 10, 30), "y": rng.normal(50, 10, 30)}
    )
    satellite = pd.DataFrame(
        {"x": rng.normal(80, 4, 5), "y": rng.normal(80, 4, 5)}
    )
    return pd.concat([base, satellite], ignore_index=True)


def fit_predict(df: pd.DataFrame, seed: int = 12345) -> pd.DataFrame:
    model = IsolationForest(
        n_estimators=200, max_samples="auto", random_state=seed
    )
    model.fit(df)
    anomaly_score = -model.score_samples(df)
    out = df.copy()
    out["anomaly_score"] = anomaly_score
    thresh = np.quantile(anomaly_score, 0.80)  # R used 0.80 here
    out["outlier"] = (anomaly_score >= thresh).astype(int)
    return out


def plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = df["outlier"].map({0: "#2c3e50", 1: "#e74c3c"})
    ax.scatter(df["x"], df["y"], c=colors, s=60, edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Experiment 3 - Cluster Anomaly\n(top 20% anomaly score flagged)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    df = make_data()
    df = fit_predict(df)

    n_flagged = int(df["outlier"].sum())
    print(f"[03] Flagged {n_flagged} / {len(df)} rows (expect the ~5 satellite + a few fringe base points).")

    out_csv = OUT_DIR / "exp3_cluster_anomalies.csv"
    df.to_csv(out_csv, index=False)
    plot(df, OUT_DIR / "exp3_cluster_anomalies.png")
    print(f"[03] Wrote {out_csv.name} + exp3_cluster_anomalies.png")


if __name__ == "__main__":
    main()
