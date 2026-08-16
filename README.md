# Edge-AutoGuard

This is an unsupervised anomaly-detection prototype built using unlabeled PHM gearbox sensor data.

The system identifies statistically unusual sensor patterns; it does not prove mechanical failure.

No physical hardware was used.

The laptop simulates the edge device.

Ground-truth fault labels were unavailable in the raw dataset.

## Overview

This prototype reads raw PHM gearbox time-series signals, divides them into windows, extracts time-domain features, trains an Isolation Forest model using unlabeled data, calculates anomaly scores, and exposes the results through a local FastAPI API and Streamlit dashboard.

The project is intentionally designed for anomaly detection rather than supervised fault classification because the raw PHM signals do not contain verified fault targets.

## Project flow

PHM09 raw sensor signals
↓
Signal preprocessing
↓
Windowing
↓
Feature extraction
↓
Isolation Forest
↓
Anomaly score
↓
Normal / Anomalous
↓
Edge inference
↓
SQLite
↓
FastAPI
↓
Streamlit dashboard

## Dataset note

The PHM09 raw dataset contains 560 CSV files, each representing a three-channel numeric signal. The files contain raw sensor values but no verified fault labels, timestamps, or healthy/failure metadata. Because of this, the prototype deliberately avoids claiming a specific mechanical fault and instead reports abnormal operating behavior using anomaly scores.

## Important terminology

Use phrases such as:

- Detected an anomalous sensor pattern.
- Detected abnormal operating behavior.
- Estimated anomaly score.

Do not claim a specific gearbox defect or failure without external verified labels.

## Structure

```text
edge-autoguard/
├── data/
├── notebooks/
├── models/
├── src/
├── api/
├── dashboard/
├── sql/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Current implementation status

- Raw PHM signal inspection complete
- Window-based feature extraction implemented
- Isolation Forest anomaly model supported
- Local edge inference simulation implemented
- SQLite prediction storage implemented
- FastAPI and Streamlit interfaces structured for anomaly results

## Notes

- This is a student-level prototype and educational project.
- The laptop is simulating the edge device locally.
- No cloud services are used.
- Results should be interpreted as statistical anomaly detection, not as validated physical-failure diagnosis.
