"""Train an Isolation Forest anomaly detector for unlabeled PHM sensor windows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.database import create_db_and_tables, insert_prediction

from src.feature_extraction import build_windowed_feature_dataset, get_feature_columns


def train_anomaly_model(
    dataset: pd.DataFrame,
    feature_columns: list[str] | None = None,
    model_path: str | Path | None = None,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
):
    """Fit an Isolation Forest to unlabeled window features and save the model bundle."""
    if feature_columns is None:
        feature_columns = get_feature_columns(dataset)

    missing = [column for column in feature_columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(dataset[feature_columns])

    bundle = {
        "model": model,
        "feature_columns": list(feature_columns),
        "contamination": contamination,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "model_type": "IsolationForest",
        "status_labels": {"normal": "Normal", "anomalous": "Anomalous"},
    }

    if model_path is not None:
        model_file = Path(model_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_file)
        feature_list_path = model_file.parent / "feature_columns.json"
        feature_list_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    return bundle


def generate_anomaly_results(dataset: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    """Return one row per processed window with anomaly scores and a status label."""
    feature_columns = model_bundle["feature_columns"]
    matrix = dataset[feature_columns].copy()
    model = model_bundle["model"]
    anom_scores = -model.score_samples(matrix)
    predictions = model.predict(matrix)
    results = dataset[["run_id", "window_id"]].copy()
    results["anomaly_score"] = anom_scores
    results["status"] = np.where(predictions == -1, "Anomalous", "Normal")
    return results


def summarize_anomaly_results(results: pd.DataFrame) -> dict:
    """Return total, run-wise, and anomaly summary values for the selected 5-run batch."""
    total_windows = int(len(results))
    normal_windows = int((results["status"] == "Normal").sum())
    anomalous_windows = int((results["status"] == "Anomalous").sum())
    anomaly_percentage = (anomalous_windows / total_windows * 100.0) if total_windows else 0.0
    run_summary = (
        results.groupby("run_id", sort=True)
        .agg(
            Total_Windows=("window_id", "count"),
            Anomalous_Windows=("status", lambda s: int((s == "Anomalous").sum())),
            Anomaly_Percent=("status", lambda s: round((s == "Anomalous").mean() * 100.0, 2)),
        )
        .reset_index()
    )
    return {
        "total_windows": total_windows,
        "normal_windows": normal_windows,
        "anomalous_windows": anomalous_windows,
        "anomaly_percentage": anomaly_percentage,
        "run_summary": run_summary,
    }


def store_anomaly_results_in_db(results: pd.DataFrame, feature_df: pd.DataFrame, db_path: str | Path, edge_device_id: str = "simulated-laptop-edge") -> None:
    """Persist batch anomaly records to the SQLite database for the selected run set."""
    db_file = Path(db_path)
    create_db_and_tables(db_file)
    engine = __import__("api.database", fromlist=["get_engine"]).get_engine(db_file)
    with engine.begin() as connection:
        connection.execute(__import__("sqlalchemy", fromlist=["text"]).text("DELETE FROM predictions"))
    feature_lookup = feature_df.set_index(["run_id", "window_id"]).to_dict(orient="index")
    for row in results.itertuples(index=False):
        key = (str(row.run_id), int(row.window_id))
        feature_data = {}
        if key in feature_lookup:
            feature_record = feature_lookup[key]
            feature_data = {k: float(v) for k, v in feature_record.items() if k not in {"run_id", "window_id", "window_size", "signal_length"}}
        insert_prediction(
            db_file,
            {
                "run_id": str(row.run_id),
                "window_id": int(row.window_id),
                "anomaly_score": float(row.anomaly_score),
                "status": str(row.status),
                "feature_data": feature_data,
                "explanation": {"top_features": []},
                "edge_device_id": edge_device_id,
                "created_at": f"run:{str(row.run_id)}-window:{row.window_id}",
            },
        )


def save_anomaly_plots(results: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate the required anomaly plots as PNG files for the 5-run experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.hist(results["anomaly_score"], bins=30, color="steelblue", edgecolor="black")
    ax1.set_title("Anomaly-score distribution")
    ax1.set_xlabel("Anomaly score")
    ax1.set_ylabel("Window count")
    fig1.tight_layout()
    fig1.savefig(output_dir / "anomaly_score_distribution.png", dpi=150)
    plt.close(fig1)

    run_counts = (
        results.groupby("run_id", sort=True)["status"]
        .apply(lambda s: int((s == "Anomalous").sum()))
        .reset_index(name="anomaly_count")
    )
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(run_counts["run_id"], run_counts["anomaly_count"], color="darkorange")
    ax2.set_title("Anomaly count by run")
    ax2.set_xlabel("Run")
    ax2.set_ylabel("Anomalous windows")
    fig2.tight_layout()
    fig2.savefig(output_dir / "anomaly_count_by_run.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(results["window_id"], results["anomaly_score"], color="forestgreen", linewidth=1.2)
    ax3.set_title("Anomaly score vs window index")
    ax3.set_xlabel("Window index")
    ax3.set_ylabel("Anomaly score")
    fig3.tight_layout()
    fig3.savefig(output_dir / "anomaly_score_vs_window_index.png", dpi=150)
    plt.close(fig3)


def print_validation_summary(results: pd.DataFrame, feature_df: pd.DataFrame, dataset_dir: Path) -> None:
    """Print the required validation output for the 5-run end-to-end anomaly analysis."""
    summary = summarize_anomaly_results(results)
    top10 = results.sort_values("anomaly_score", ascending=False).head(10)
    most_anomalous = top10.iloc[0]
    raw_df = pd.read_csv(dataset_dir / f"{most_anomalous['run_id']}.csv", header=None)
    window_start = int(most_anomalous["window_id"]) * 1024
    window_stop = window_start + 1024
    raw_signals = {
        "Signal_1": raw_df.iloc[:, 0].iloc[window_start:window_stop].tolist(),
        "Signal_2": raw_df.iloc[:, 1].iloc[window_start:window_stop].tolist(),
        "Signal_3": raw_df.iloc[:, 2].iloc[window_start:window_stop].tolist(),
    }
    feature_row = feature_df[(feature_df["run_id"] == most_anomalous["run_id"]) & (feature_df["window_id"] == most_anomalous["window_id"])].iloc[0]
    feature_values = {key: float(value) for key, value in feature_row.items() if key not in {"run_id", "window_id", "window_size", "signal_length"}}

    print("1. Number of runs processed:", results["run_id"].nunique())
    print("2. Number of windows generated:", len(results))
    print("3. Number of normal windows:", summary["normal_windows"])
    print("4. Number of anomalous windows:", summary["anomalous_windows"])
    print("5. Anomaly percentage:", round(summary["anomaly_percentage"], 2))
    print("6. Anomalies per run:")
    print(summary["run_summary"].to_string(index=False))
    print("7. Top 10 anomalous windows:")
    print(top10.to_string(index=False))
    print("8. Feature values of the most anomalous window:")
    print("run_id:", most_anomalous["run_id"])
    print("window_id:", most_anomalous["window_id"])
    print("anomaly_score:", most_anomalous["anomaly_score"])
    print("features:")
    for key, value in feature_values.items():
        print(f"  {key}: {value}")
    print("raw Signal_1:", raw_signals["Signal_1"])
    print("raw Signal_2:", raw_signals["Signal_2"])
    print("raw Signal_3:", raw_signals["Signal_3"])


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root.parent / "PHM09_competition_1"
    selected_run_ids = [f"Run_{run_number}" for run_number in range(1, 6)]
    feature_df = build_windowed_feature_dataset(
        dataset_dir,
        window_size=1024,
        stride=1024,
        run_ids=selected_run_ids,
    )
    feature_path = project_root / "data" / "extracted_features.csv"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(feature_path, index=False)

    feature_columns = get_feature_columns(feature_df)
    model_bundle = train_anomaly_model(
        feature_df,
        feature_columns=feature_columns,
        model_path=project_root / "models" / "anomaly_model.pkl",
        contamination=0.05,
        n_estimators=200,
        random_state=42,
    )

    anomaly_results = generate_anomaly_results(feature_df, model_bundle)
    results_path = project_root / "data" / "anomaly_results.csv"
    anomaly_results.to_csv(results_path, index=False)

    db_path = project_root / "data" / "edge_autoguard.db"
    store_anomaly_results_in_db(anomaly_results, feature_df, db_path)
    plot_dir = project_root / "data" / "plots"
    save_anomaly_plots(anomaly_results, plot_dir)

    summary = summarize_anomaly_results(anomaly_results)
    total_windows = summary["total_windows"]
    normal_windows = summary["normal_windows"]
    anomalous_windows = summary["anomalous_windows"]
    anomaly_percentage = summary["anomaly_percentage"]

    print("Run set:", selected_run_ids)
    print("Total windows:", total_windows)
    print("Normal windows:", normal_windows)
    print("Anomalous windows:", anomalous_windows)
    print("Anomaly percentage:", round(anomaly_percentage, 2), "%")
    print("Anomalies per run:\n", summary["run_summary"].to_string(index=False))
    print("Top 10 anomalous windows:\n", anomaly_results.sort_values("anomaly_score", ascending=False).head(10).to_string(index=False))
    print("Feature dataset saved:", feature_path)
    print("Anomaly results saved:", results_path)
    print("Saved model:", project_root / "models" / "anomaly_model.pkl")
    print("Saved plots:", plot_dir)
    print("Database:", db_path)
    print_validation_summary(anomaly_results, feature_df, dataset_dir)
