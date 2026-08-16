from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(nullable=False)
    window_id: Mapped[int] = mapped_column(nullable=False)
    anomaly_score: Mapped[float] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(nullable=False)
    feature_data: Mapped[str] = mapped_column(nullable=False)
    explanation: Mapped[str] = mapped_column(nullable=False)
    edge_device_id: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[str] = mapped_column(nullable=False)


def get_engine(db_path: str | Path = "data/edge_autoguard.db"):
    path = str(Path(db_path))
    engine = create_engine(f"sqlite:///{path}")
    return engine


def create_db_and_tables(db_path: str | Path = "data/edge_autoguard.db") -> str:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(db_file)
    with engine.begin() as connection:
        inspector = inspect(connection)
        if "predictions" in inspector.get_table_names():
            existing_columns = {column["name"] for column in inspector.get_columns("predictions")}
            expected_columns = {"id", "run_id", "window_id", "anomaly_score", "status", "feature_data", "explanation", "edge_device_id", "created_at"}
            if not expected_columns.issubset(existing_columns):
                connection.execute(text("DROP TABLE predictions"))
    Base.metadata.create_all(engine)
    return str(db_file)


def insert_prediction(db_path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    db_file = create_db_and_tables(db_path)
    engine = get_engine(db_file)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        row = PredictionRecord(
            run_id=str(record["run_id"]),
            window_id=int(record["window_id"]),
            anomaly_score=float(record["anomaly_score"]),
            status=str(record["status"]),
            feature_data=json.dumps(record.get("feature_data", {}), default=str),
            explanation=json.dumps(record.get("explanation", {}), default=str),
            edge_device_id=str(record.get("edge_device_id", "unknown-edge")),
            created_at=str(record.get("created_at") or f"run:{record['run_id']}-window:{record['window_id']}"),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return {
            "id": row.id,
            "run_id": row.run_id,
            "window_id": row.window_id,
            "anomaly_score": row.anomaly_score,
            "status": row.status,
            "feature_data": json.loads(row.feature_data),
            "explanation": json.loads(row.explanation),
            "edge_device_id": row.edge_device_id,
            "created_at": row.created_at,
        }


def fetch_predictions(db_path: str | Path, run_id: str | None = None) -> list[dict[str, Any]]:
    db_file = create_db_and_tables(db_path)
    engine = get_engine(db_file)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        query = text(
            "SELECT id, run_id, window_id, anomaly_score, status, feature_data, explanation, edge_device_id, created_at FROM predictions"
        )
        if run_id is not None:
            query = text(
                "SELECT id, run_id, window_id, anomaly_score, status, feature_data, explanation, edge_device_id, created_at FROM predictions WHERE run_id = :run_id"
            )
            rows = session.execute(query, {"run_id": run_id}).fetchall()
        else:
            rows = session.execute(query).fetchall()

        result = []
        for row in rows:
            result.append(
                {
                    "id": row[0],
                    "run_id": row[1],
                    "window_id": row[2],
                    "anomaly_score": row[3],
                    "status": row[4],
                    "feature_data": json.loads(row[5]),
                    "explanation": json.loads(row[6]),
                    "edge_device_id": row[7],
                    "created_at": row[8],
                }
            )
        return result
