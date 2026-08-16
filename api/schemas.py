from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionCreate(BaseModel):
    run_id: str = Field(..., min_length=1)
    window_id: int = Field(..., ge=0)
    anomaly_score: float = Field(...)
    status: str = Field(..., min_length=1)
    feature_data: dict[str, Any] = Field(default_factory=dict)
    explanation: dict[str, Any] = Field(default_factory=dict)
    edge_device_id: str = Field(..., min_length=1)
    created_at: str | None = None


class PredictionOut(PredictionCreate):
    id: int
