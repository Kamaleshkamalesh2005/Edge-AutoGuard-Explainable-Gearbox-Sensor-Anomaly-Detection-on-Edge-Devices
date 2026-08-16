"""Inspect the PHM gearbox dataset and print a concise summary."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd


def inspect_dataset(data_dir: str | os.PathLike[str]) -> dict:
    """Summarize the raw PHM CSV run files."""
    dataset_dir = Path(data_dir)
    files = sorted(dataset_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    sample = pd.read_csv(files[0], header=None)
    shape_counts: dict[tuple[int, int], int] = {}
    for file in files:
        df = pd.read_csv(file, header=None)
        shape_counts[df.shape] = shape_counts.get(df.shape, 0) + 1

    combined = pd.concat([pd.read_csv(file, header=None) for file in files[:20]], ignore_index=True)
    summary = {
        "file_count": len(files),
        "sample_shape": sample.shape,
        "unique_shapes": sorted(shape_counts.items()),
        "sample_rows": sample.head(5).values.tolist(),
        "column_means": combined.mean().round(6).tolist(),
        "column_std": combined.std().round(6).tolist(),
        "min_max": {
            str(i): {
                "min": float(combined.iloc[:, i].min()),
                "max": float(combined.iloc[:, i].max()),
            }
            for i in range(combined.shape[1])
        },
    }
    return summary


if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parents[1] / ".." / "PHM09_competition_1"
    stats = inspect_dataset(dataset_dir)
    print("PHM dataset inspection summary")
    print(f"CSV files: {stats['file_count']}")
    print(f"Sample file shape: {stats['sample_shape']}")
    print("First rows:")
    for row in stats["sample_rows"]:
        print(row)
    print("Column means:", stats["column_means"])
    print("Column stddev:", stats["column_std"])
    print("Min/max by column:", stats["min_max"])
