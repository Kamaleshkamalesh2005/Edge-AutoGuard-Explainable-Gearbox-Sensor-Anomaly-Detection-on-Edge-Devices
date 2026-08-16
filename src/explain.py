"""SHAP explanation utilities for the trained gearbox classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import shap


def explain_prediction(model_bundle: dict, feature_row: list[float], feature_columns: list[str] | None = None):
    """Return a dictionary with SHAP values and feature importances for one prediction."""
    model = model_bundle["model"]
    feature_list = feature_columns if feature_columns is not None else model_bundle["feature_columns"]
    array = np.asarray(feature_row, dtype=float).reshape(1, -1)
    if array.shape[1] != len(feature_list):
        raise ValueError(f"Expected {len(feature_list)} features but got {array.shape[1]}.")

    explainer = shap.Explainer(model, feature_names=feature_list)
    shap_values = explainer(array)

    raw_predicted = int(model.predict(array)[0])
    reverse_map = {0: "healthy", 1: "fault"}
    predicted_class = reverse_map.get(raw_predicted, str(raw_predicted))
    probas = model.predict_proba(array)[0]
    class_index = 1 if len(model.classes_) > 1 and 1 in model.classes_ else 0
    probability = float(probas[class_index]) if class_index < len(probas) else float(probas[-1])

    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = values[0, :, :]
    if values.ndim == 2:
        values = values[0]

    ordered = sorted(
        zip(feature_list, values.tolist()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    top_features = [{"feature": name, "shap_value": float(value)} for name, value in ordered[:5]]

    return {
        "predicted_class": str(predicted_class),
        "probability": probability,
        "top_features": top_features,
        "shap_values": {feature: float(value) for feature, value in ordered},
    }


if __name__ == "__main__":
    model_path = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    bundle = joblib.load(model_path)
    sample_row = [0.5, 0.1, 0.0, 0.3, 0.2, 0.1, 0.5, 0.01, 0.2]
    result = explain_prediction(bundle, sample_row)
    print(json.dumps(result, indent=2))
