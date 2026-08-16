"""Evaluation utilities for the gearbox classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(model_bundle: dict, test_df: pd.DataFrame, feature_columns: list[str]):
    """Evaluate the model and print performance metrics for the test split."""
    X_test = test_df[feature_columns]
    y_test = test_df["condition"]
    model = model_bundle["model"]
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=model.classes_).tolist(),
        "classification_report": classification_report(y_test, y_pred, digits=3, zero_division=0),
    }
    return metrics


if __name__ == "__main__":
    model_path = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    bundle = joblib.load(model_path)
    feature_columns = bundle["feature_columns"]
    print("Model classes:", bundle["classes"])
    print("Feature count:", len(feature_columns))
