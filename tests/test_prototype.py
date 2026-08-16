from pathlib import Path

import pandas as pd
import pytest

from api.database import create_db_and_tables, fetch_predictions, insert_prediction
from api.schemas import PredictionCreate
from src.feature_extraction import extract_signal_features
from src.predict import predict_anomaly_from_features
from src.train import train_anomaly_model


@pytest.fixture
def sample_signal():
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.2, -0.1, 0.0, 0.3, 0.5]


def test_extract_signal_features_returns_expected_keys(sample_signal):
    features = extract_signal_features(sample_signal)
    assert "mean" in features
    assert "std" in features
    assert "rms" in features
    assert "peak_to_peak" in features
    assert "kurtosis" in features
    assert "skewness" in features
    assert features["mean"] > 0


def test_window_feature_dataset_is_created_without_labels(tmp_path):
    csv_dir = tmp_path / "runs"
    csv_dir.mkdir()

    low_signal = [0.1, 0.0, -0.1, 0.05, 0.0, -0.04, 0.02, 0.01] * 3
    high_signal = [1.2, 0.8, -1.0, 1.5, 0.9, -1.3, 1.1, 0.7] * 3
    df_low = pd.DataFrame([low_signal, low_signal])
    df_high = pd.DataFrame([high_signal, high_signal])
    df_low.to_csv(csv_dir / "Run_1.csv", header=False, index=False)
    df_high.to_csv(csv_dir / "Run_2.csv", header=False, index=False)

    dataset = pd.concat([
        pd.DataFrame({"run_id": ["Run_1", "Run_1"], "window_id": [0, 1], "window_size": [8, 8], "signal_length": [8, 8], "mean_Signal_1": [0.1, 0.1], "std_Signal_1": [0.05, 0.05]}),
        pd.DataFrame({"run_id": ["Run_2", "Run_2"], "window_id": [0, 1], "window_size": [8, 8], "signal_length": [8, 8], "mean_Signal_1": [1.2, 1.2], "std_Signal_1": [0.4, 0.4]}),
    ], ignore_index=True)

    assert "run_id" in dataset.columns
    assert "window_id" in dataset.columns
    assert "mean_Signal_1" in dataset.columns


def test_train_anomaly_model_and_predict(tmp_path):
    data = pd.DataFrame(
        {
            "rms_Signal_1": [0.4, 0.5, 0.6, 1.2, 1.5, 1.4],
            "std_Signal_1": [0.1, 0.2, 0.3, 0.8, 1.0, 0.9],
            "mean_Signal_1": [0.0, 0.1, 0.2, 1.0, 1.2, 1.1],
        }
    )
    model_path = tmp_path / "anomaly_model.pkl"
    feature_columns = ["rms_Signal_1", "std_Signal_1", "mean_Signal_1"]
    model = train_anomaly_model(data, feature_columns=feature_columns, model_path=str(model_path), contamination=0.2)
    statuses, scores = predict_anomaly_from_features(model, [[0.5, 0.2, 0.1]], feature_columns)
    assert statuses[0] in {"Normal", "Anomalous"}
    assert isinstance(scores[0], float)
    assert model is not None


def test_prediction_schema_validates_data():
    payload = {
        "run_id": "Run_12",
        "window_id": 3,
        "anomaly_score": 0.87,
        "status": "Anomalous",
        "feature_data": {"rms_Signal_1": 1.2, "std_Signal_1": 0.8},
        "explanation": {"top_features": [{"feature": "rms_Signal_1", "score": 0.31}]},
        "edge_device_id": "edge-laptop",
    }
    model = PredictionCreate(**payload)
    assert model.run_id == "Run_12"
    assert model.status == "Anomalous"


def test_database_insert_and_fetch(tmp_path):
    db_path = tmp_path / "test.db"
    create_db_and_tables(str(db_path))
    record = {
        "run_id": "Run_99",
        "window_id": 1,
        "anomaly_score": 0.91,
        "status": "Anomalous",
        "feature_data": {"rms_Signal_1": 1.2},
        "explanation": {"top_features": [{"feature": "rms_Signal_1", "score": 0.32}]},
        "edge_device_id": "edge-laptop",
    }
    inserted = insert_prediction(str(db_path), record)
    rows = fetch_predictions(str(db_path), run_id="Run_99")
    assert inserted is not None
    assert len(rows) >= 1
    assert rows[0]["status"] == "Anomalous"


def test_window_feature_dataset_uses_numeric_run_order(tmp_path):
    csv_dir = tmp_path / "runs"
    csv_dir.mkdir()

    for run_num in [1, 2, 3, 4, 5, 10]:
        rows = [[float(run_num), float(run_num), float(run_num)] for _ in range(2048)]
        pd.DataFrame(rows).to_csv(csv_dir / f"Run_{run_num}.csv", header=False, index=False)

    dataset = pd.read_csv(csv_dir / "Run_1.csv", header=None)
    assert dataset.shape == (2048, 3)

    feature_dataset = __import__('src.feature_extraction', fromlist=['build_windowed_feature_dataset']).build_windowed_feature_dataset(
        csv_dir,
        window_size=1024,
        stride=1024,
        max_runs=5,
    )

    run_order = feature_dataset["run_id"].unique().tolist()
    assert run_order == ["Run_1", "Run_2", "Run_3", "Run_4", "Run_5"]


def test_sql_queries_file_contains_required_queries():
    query_path = Path(__file__).resolve().parents[1] / "sql" / "queries.sql"
    text = query_path.read_text(encoding="utf-8")
    for keyword in ["run_id", "window_id", "COUNT", "AVG", "latest"]:
        assert keyword.lower() in text.lower()
