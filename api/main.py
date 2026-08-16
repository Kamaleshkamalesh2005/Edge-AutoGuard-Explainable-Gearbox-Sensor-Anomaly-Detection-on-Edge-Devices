from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.crud import create_prediction_record, get_prediction_history
from api.database import create_db_and_tables
from api.schemas import PredictionCreate

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "edge_autoguard.db"

app = FastAPI(title="Edge-AutoGuard API", version="0.1.0")
create_db_and_tables(DB_PATH)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "database": str(DB_PATH), "model_type": "IsolationForest"}


@app.post("/api/predictions")
def create_prediction(payload: PredictionCreate):
    try:
        record = create_prediction_record(DB_PATH, payload.model_dump())
        return {"status": "success", "prediction": record}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to store prediction: {exc}") from exc


@app.get("/api/predictions")
def list_predictions():
    try:
        records = get_prediction_history(DB_PATH)
        return {"count": len(records), "predictions": records}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Database failure: {exc}") from exc


@app.get("/api/predictions/{run_id}")
def get_run_predictions(run_id: str):
    try:
        records = get_prediction_history(DB_PATH, run_id=run_id)
        return {"run_id": run_id, "count": len(records), "predictions": records}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Database failure: {exc}") from exc
