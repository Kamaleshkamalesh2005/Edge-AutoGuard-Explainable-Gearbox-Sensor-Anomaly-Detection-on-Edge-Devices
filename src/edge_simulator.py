"""Simulate local edge-device inference for unlabeled gearbox sensor anomaly detection."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from src.feature_extraction import extract_feature_row, window_signal


def load_signal_from_csv(csv_path: str | Path):
    """Load a raw PHM sensor signal CSV from disk."""
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 3:
        raise ValueError(f"The CSV file must have at least 3 columns: {csv_path}")
    return df.iloc[:, :3].to_numpy(dtype=float)


def _top_feature_contributions(feature_dict: dict[str, float], feature_columns: list[str]) -> list[dict[str, float | str]]:
    """Use feature magnitude relative to the feature distribution to summarize localized abnormality."""
    values = []
    for name in feature_columns:
        if name in feature_dict:
            values.append((name, abs(float(feature_dict[name]))))
    values.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "score": float(value)} for name, value in values[:5]]


def run_edge_prediction(csv_path: str | Path, model_path: str | Path, api_url: str = "http://127.0.0.1:8000/api/predictions"):
    """Run local inference on a single signal window and send the anomaly result to the FastAPI API."""
    model_bundle = joblib.load(model_path)
    feature_columns = model_bundle["feature_columns"]

    signal = load_signal_from_csv(csv_path)
    signal_window = window_signal(signal[:, 0], window_size=1024, stride=1024)[0]
    window_signals = [signal[:, index][: len(signal_window)] for index in range(signal.shape[1])]
    feature_dict = extract_feature_row(window_signals, ["Signal_1", "Signal_2", "Signal_3"])
    ordered_features = [float(feature_dict.get(column, 0.0)) for column in feature_columns if column in feature_dict]

    model = model_bundle["model"]
    anomaly_score = float(-model.score_samples(np.asarray([ordered_features], dtype=float))[0])
    prediction = int(model.predict(np.asarray([ordered_features], dtype=float))[0])
    status = "Anomalous" if prediction == -1 else "Normal"

    explanation = {
        "status": status,
        "anomaly_score": anomaly_score,
        "top_features": _top_feature_contributions(feature_dict, feature_columns),
        "raw_signal": [float(value) for value in signal_window.tolist()],
    }

    payload = {
        "run_id": Path(csv_path).stem,
        "window_id": 0,
        "anomaly_score": anomaly_score,
        "status": status,
        "feature_data": {
            **{key: float(value) for key, value in feature_dict.items()},
            "raw_signal": [float(value) for value in signal_window.tolist()],
        },
        "explanation": explanation,
        "edge_device_id": "simulated-laptop-edge",
    }

    response = requests.post(api_url, json=payload, timeout=10)
    if response.status_code >= 400:
        raise RuntimeError(f"API request failed with {response.status_code}: {response.text}")

    response_json = response.json()
    print("Edge anomaly check completed.")
    print(f"Status: {status}")
    print(f"Anomaly score: {anomaly_score:.4f}")
    print("Top features:", explanation["top_features"][:3])
    return response_json


if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parents[1].parent / "PHM09_competition_1"
    sample_file = sorted(Path(dataset_dir).glob("*.csv"))[0]
    model_file = Path(__file__).resolve().parents[1] / "models" / "anomaly_model.pkl"
    run_edge_prediction(sample_file, model_file)
