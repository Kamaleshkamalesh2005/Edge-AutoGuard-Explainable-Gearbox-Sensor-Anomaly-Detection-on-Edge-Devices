from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


def find_dataset_root(project_root: Path) -> Path:
    candidates = [
        project_root / "PHM09_competition_1",
        project_root.parent / "PHM09_competition_1",
        Path.cwd() / "PHM09_competition_1",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate PHM09_competition_1 relative to the project. "
        "Please confirm the dataset folder path and update the script input."
    )


def summarize_file(file_path: Path) -> dict:
    df = pd.read_csv(file_path, header=None)
    row_count = int(df.shape[0])
    col_count = int(df.shape[1])

    column_names = [f"Column_{index + 1}" for index in range(col_count)]
    dtypes = {name: str(dtype) for name, dtype in zip(column_names, df.dtypes)}
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    stats = df.describe(include='all').transpose().to_dict()

    low_cardinality = {}
    for idx, col_name in enumerate(column_names):
        series = df.iloc[:, idx]
        unique_count = int(series.nunique(dropna=True))
        if unique_count <= 10:
            low_cardinality[col_name] = {
                "unique_count": unique_count,
                "values": [value for value in series.dropna().unique().tolist()[:10]],
            }

    file_summary = {
        "filename": file_path.name,
        "rows": row_count,
        "columns": col_count,
        "column_names": column_names,
        "dtypes": dtypes,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "first_5_rows": df.head(5).values.tolist(),
        "basic_statistics": stats,
        "low_cardinality_columns": low_cardinality,
        "contains_timestamps": False,
        "contains_sensor_measurements": True,
        "contains_operating_conditions": False,
        "contains_rpm_or_speed": False,
        "contains_load": False,
        "contains_labels": False,
        "contains_fault_or_healthy_info": False,
        "contains_experiment_or_run_ids": False,
    }

    return file_summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = find_dataset_root(project_root)
    files = sorted(dataset_root.glob("Run_*.csv"))
    if not files:
        raise FileNotFoundError(f"No Run_*.csv files found in {dataset_root}")

    summaries = [summarize_file(file_path) for file_path in files]

    summary_path = project_root / "data" / "dataset_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "rows",
        "columns",
        "column_names",
        "dtypes",
        "missing_values",
        "duplicate_rows",
        "contains_timestamps",
        "contains_sensor_measurements",
        "contains_operating_conditions",
        "contains_rpm_or_speed",
        "contains_load",
        "contains_labels",
        "contains_fault_or_healthy_info",
        "contains_experiment_or_run_ids",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow({
                "filename": item["filename"],
                "rows": item["rows"],
                "columns": item["columns"],
                "column_names": json.dumps(item["column_names"]),
                "dtypes": json.dumps(item["dtypes"]),
                "missing_values": item["missing_values"],
                "duplicate_rows": item["duplicate_rows"],
                "contains_timestamps": item["contains_timestamps"],
                "contains_sensor_measurements": item["contains_sensor_measurements"],
                "contains_operating_conditions": item["contains_operating_conditions"],
                "contains_rpm_or_speed": item["contains_rpm_or_speed"],
                "contains_load": item["contains_load"],
                "contains_labels": item["contains_labels"],
                "contains_fault_or_healthy_info": item["contains_fault_or_healthy_info"],
                "contains_experiment_or_run_ids": item["contains_experiment_or_run_ids"],
            })

    total_rows = sum(item["rows"] for item in summaries)
    print("DATASET_ROOT:", dataset_root)
    print("CSV_FILES:", len(files))
    print("TOTAL_ROWS:", total_rows)
    print("SUMMARY_PATH:", summary_path)
    print("SAMPLE_FILES:", [item["filename"] for item in summaries[:5]])
    print("SAMPLE_SHAPES:", {item["filename"]: (item["rows"], item["columns"]) for item in summaries[:5]})
    print("SAMPLE_FIRST_ROWS:", summaries[0]["first_5_rows"])


if __name__ == "__main__":
    main()
