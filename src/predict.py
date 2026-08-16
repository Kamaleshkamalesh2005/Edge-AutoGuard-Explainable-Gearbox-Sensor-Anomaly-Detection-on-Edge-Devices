"""Run local anomaly detection on feature rows with the trained Isolation Forest."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _load_model_bundle(model_path: str | Path):
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError("The saved model file is invalid or missing required keys.")
    return bundle


def predict_anomaly_from_features(model_bundle: dict, feature_rows: list[list[float]], feature_columns: list[str] | None = None):
    """Return status labels and anomaly scores for one or more feature rows."""
    model = model_bundle["model"]
    feature_list = feature_columns if feature_columns is not None else model_bundle["feature_columns"]
    features = np.asarray(feature_rows, dtype=float)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.shape[1] != len(feature_list):
        raise ValueError(f"Expected {len(feature_list)} features but got {features.shape[1]}.")

    frame = pd.DataFrame(features, columns=feature_list)
    predictions = model.predict(frame)
    scores = -model.score_samples(frame)
    statuses = ["Anomalous" if decision == -1 else "Normal" for decision in predictions]
    return statuses, scores.tolist()


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "models" / "anomaly_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    bundle = _load_model_bundle(path)
    sample = [[0.23, 0.14, 0.05, 0.32, 0.21, 0.18, 0.47, 0.02, 0.11] for _ in range(1)]
    statuses, scores = predict_anomaly_from_features(bundle, sample)
    print("Status:", statuses)
    print("Anomaly score:", scores)
