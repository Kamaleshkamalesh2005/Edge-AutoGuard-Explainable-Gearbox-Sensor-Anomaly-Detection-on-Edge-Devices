from __future__ import annotations

from pathlib import Path
from typing import Any

from api.database import fetch_predictions, insert_prediction


def create_prediction_record(db_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    record = {
        "run_id": payload["run_id"],
        "window_id": payload["window_id"],
        "anomaly_score": payload["anomaly_score"],
        "status": payload["status"],
        "feature_data": payload.get("feature_data", {}),
        "explanation": payload.get("explanation", {}),
        "edge_device_id": payload["edge_device_id"],
        "created_at": payload.get("created_at") or f"run:{payload['run_id']}-window:{payload['window_id']}",
    }
    return insert_prediction(db_path, record)


def get_prediction_history(db_path: str | Path, run_id: str | None = None) -> list[dict[str, Any]]:
    return fetch_predictions(db_path, run_id=run_id)
