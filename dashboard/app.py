from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from api.database import fetch_predictions

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "edge_autoguard.db"

st.set_page_config(page_title="Edge-AutoGuard Dashboard", layout="wide")
st.title("Edge-AutoGuard — Explainable Gearbox Sensor Anomaly Detection")
st.caption("Unsupervised anomaly detection using unlabeled PHM gearbox sensor windows. The laptop simulates the edge device.")

records = fetch_predictions(DB_PATH)
if not records:
    st.info("No anomaly predictions available yet. Run the edge simulator or submit a prediction through the API.")
    st.stop()

history_df = pd.DataFrame(records)
latest = history_df.iloc[-1]

st.subheader("Current anomaly summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Run", str(latest["run_id"]))
col2.metric("Window", int(latest["window_id"]))
col3.metric("Status", str(latest["status"]))
col4.metric("Anomaly Score", f"{float(latest['anomaly_score']):.4f}")

st.subheader("Latest window")
latest_df = history_df[["run_id", "window_id", "status", "anomaly_score", "created_at"]].copy()
st.dataframe(latest_df, use_container_width=True)

feature_data = latest.get("feature_data", {})
if isinstance(feature_data, dict) and feature_data:
    st.subheader("Feature summary")
    st.json({k: float(v) if isinstance(v, (int, float)) else v for k, v in feature_data.items() if not isinstance(v, list)})

explanation = latest.get("explanation", {})
if explanation:
    st.subheader("Top abnormal features")
    top_features = explanation.get("top_features", [])
    if top_features:
        feature_df = pd.DataFrame(top_features)
        st.dataframe(feature_df, use_container_width=True)
        fig = px.bar(feature_df, x="feature", y="score", color="score", title="Abnormal feature magnitude")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No abnormal-feature explanation available.")

st.subheader("Anomaly analytics")
row_count = len(history_df)
normal_count = int((history_df["status"].str.lower() == "normal").sum())
anomalous_count = int((history_df["status"].str.lower() == "anomalous").sum())
anomaly_percentage = (anomalous_count / row_count * 100.0) if row_count else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total windows", row_count)
c2.metric("Normal windows", normal_count)
c3.metric("Anomalous windows", anomalous_count)
c4.metric("Anomaly %", f"{anomaly_percentage:.2f}%")

score_hist = px.histogram(history_df, x="anomaly_score", nbins=30, title="Anomaly score distribution")
st.plotly_chart(score_hist, use_container_width=True)

run_counts = history_df.groupby("run_id")["status"].apply(lambda s: (s.str.lower() == "anomalous").sum()).reset_index(name="anomaly_count")
run_plot = px.bar(run_counts, x="run_id", y="anomaly_count", title="Number of anomalies by Run")
st.plotly_chart(run_plot, use_container_width=True)

score_vs_window = px.line(
    history_df[["window_id", "anomaly_score"]].sort_values("window_id"),
    x="window_id",
    y="anomaly_score",
    title="Anomaly Score vs Window Index",
)
st.plotly_chart(score_vs_window, use_container_width=True)

if "raw_signal" in str(feature_data):
    raw_signal = feature_data.get("raw_signal", [])
    if raw_signal:
        signal_df = pd.DataFrame({
            "Sample Index": list(range(len(raw_signal))),
            "Signal_1": raw_signal[:len(raw_signal)],
            "Signal_2": raw_signal[:len(raw_signal)],
            "Signal_3": raw_signal[:len(raw_signal)],
        })
        st.line_chart(signal_df[["Signal_1", "Signal_2", "Signal_3"]])

raw_signal = explanation.get("raw_signal", [])
if raw_signal:
    st.subheader("Selected raw signal window")
    signal_df = pd.DataFrame({
        "Sample Index": list(range(len(raw_signal))),
        "Signal_1": raw_signal,
    })
    st.line_chart(signal_df.set_index("Sample Index"))
