"""
Anomaly Detection -- EXPERIMENT 1
=============================================
METHOD : scikit-learn IsolationForest
CASE   : A single point lifted far above a tight Gaussian blob.
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

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 2.0 GENERATE DATA -- deliberately planted single outlier                    #
# --------------------------------------------------------------------------- #
def make_data(seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(50, 10, size=30)
    y = rng.normal(50, 10, size=30)
    y[0] = y[0] + 40
    return pd.DataFrame({"x": x, "y": y})


# --------------------------------------------------------------------------- #
# 3.0 ISOLATION FOREST                                                        #
# --------------------------------------------------------------------------- #
def fit_iforest(df: pd.DataFrame, seed: int = 12345) -> IsolationForest:
    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        max_features=1.0,
        # h2o's "predict" score is higher = more anomalous; sklearn's
        # score_samples is higher = more normal. We keep sklearn's
        # convention and flip signs where needed.
        contamination="auto",
        random_state=seed,
    )
    model.fit(df)
    return model


def score(df: pd.DataFrame, model: IsolationForest) -> pd.DataFrame:
    # Larger = more anomalous,.
    anomaly_score = -model.score_samples(df)
    out = df.copy()
    out["anomaly_score"] = anomaly_score
    thresh = np.quantile(anomaly_score, 0.99)
    out["outlier"] = (anomaly_score >= thresh).astype(int)
    return out


# --------------------------------------------------------------------------- #
# 5.0 VIZ                                                                     #
# --------------------------------------------------------------------------- #
def plot(out_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = out_df["outlier"].map({0: "#2c3e50", 1: "#e74c3c"})
    ax.scatter(out_df["x"], out_df["y"], c=colors, s=60, edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Experiment 1 - Point Anomaly\n(Isolation Forest, top 1% anomaly score)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    df = make_data()
    model = fit_iforest(df)
    out = score(df, model)

    print("[01] Top 5 by anomaly score:")
    print(out.sort_values("anomaly_score", ascending=False).head().to_string(index=False))

    out_csv = OUT_DIR / "exp1_point_anomalies.csv"
    out.to_csv(out_csv, index=False)

    plot(out, OUT_DIR / "exp1_point_anomalies.png")
    print(f"[01] Wrote {out_csv.name} and exp1_point_anomalies.png")


if __name__ == "__main__":
    main()

# %%
