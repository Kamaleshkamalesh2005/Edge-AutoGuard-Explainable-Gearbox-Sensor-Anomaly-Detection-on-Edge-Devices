"""Data loading and preprocessing for the PHM gearbox dataset."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_extraction import extract_feature_row


def load_signal_runs(data_dir: str | os.PathLike[str]) -> list[tuple[str, pd.DataFrame]]:
    """Load each raw run as a DataFrame and keep the file name."""
    directory = Path(data_dir)
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    runs: list[tuple[str, pd.DataFrame]] = []
    for file in files:
        df = pd.read_csv(file, header=None)
        if df.shape[1] < 3:
            raise ValueError(f"Unexpected column count in {file.name}: expected at least 3 but found {df.shape[1]}")
        runs.append((file.stem, df.iloc[:, :3].copy()))
    return runs


def build_feature_dataset(data_dir: str | os.PathLike[str]) -> pd.DataFrame:
    """Convert each raw run into a row of features and create a proxy condition label."""
    rows: list[dict] = []
    for run_id, df in load_signal_runs(data_dir):
        signal_columns = [df.iloc[:, i].to_numpy(dtype=float) for i in range(min(3, df.shape[1]))]
        feature_row = extract_feature_row(signal_columns)
        feature_row["run_id"] = run_id
        rows.append(feature_row)

    if not rows:
        raise ValueError(f"No feature rows could be created from {data_dir}")

    dataset = pd.DataFrame(rows)
    if "sensor_1_rms" not in dataset.columns:
        raise ValueError("The feature dataset does not contain the expected sensor RMS features.")

    sensor_rms_cols = [column for column in dataset.columns if column.endswith("_rms")]
    dataset["signal_energy"] = dataset[sensor_rms_cols].sum(axis=1)

    # Keep the detailed sensor-level features and add a compact summary for model simplicity.
    dataset["mean"] = dataset[[column for column in dataset.columns if column.endswith("_mean")]].mean(axis=1)
    dataset["std"] = dataset[[column for column in dataset.columns if column.endswith("_std")]].mean(axis=1)
    dataset["rms"] = dataset[sensor_rms_cols].mean(axis=1)
    dataset["peak_to_peak"] = dataset[[column for column in dataset.columns if column.endswith("_peak_to_peak")]].mean(axis=1)
    dataset["kurtosis"] = dataset[[column for column in dataset.columns if column.endswith("_kurtosis")]].mean(axis=1)
    dataset["skewness"] = dataset[[column for column in dataset.columns if column.endswith("_skewness")]].mean(axis=1)

    threshold = dataset["signal_energy"].median() + 0.5 * dataset["signal_energy"].std(ddof=0)
    dataset["condition"] = dataset["signal_energy"].apply(lambda value: "fault" if value > threshold else "healthy")
    return dataset


def train_test_split_by_run(dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split by run ID so one experiment stays entirely in one split."""
    if "run_id" not in dataset.columns:
        raise ValueError("The dataset must include a run_id before splitting.")
    if "condition" not in dataset.columns:
        raise ValueError("The dataset must include a condition label before splitting.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(dataset, dataset["condition"], groups=dataset["run_id"]))
    train_df = dataset.iloc[train_idx].reset_index(drop=True)
    test_df = dataset.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / ".." / "PHM09_competition_1"
    feature_df = build_feature_dataset(data_dir)
    print(feature_df.head())
    print("Condition counts:\n", feature_df["condition"].value_counts())
