"""
Run every stage of the Python port end-to-end:

    00_data_prep.py
    01_experiment_1_point_anomalies.py
    02_experiment_2_timeseries_anomalies.py
    03_experiment_3_cluster_anomalies.py
    04_fraud_anomaly_detection.py
    05_feature_discrimination.py
    06_temporal_validation.py

Usage:  python run_all.py
"""

from __future__ import annotations

import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    "00_data_prep.py",
    "01_experiment_1_point_anomalies.py",
    "02_experiment_2_timeseries_anomalies.py",
    "03_experiment_3_cluster_anomalies.py",
    "04_fraud_anomaly_detection.py",
    "05_feature_discrimination.py",
    "06_temporal_validation.py",
]


def main() -> None:
    for stage in STAGES:
        path = HERE / stage
        print(f"\n==== running {stage} " + "=" * (60 - len(stage)))
        runpy.run_path(str(path), run_name="__main__")
    print("\n==== all stages complete ====")


if __name__ == "__main__":
    main()
