# 🚗 Edge-AutoGuard

### Explainable Gearbox Sensor Anomaly Detection on Edge Devices

> **An AI-powered predictive-maintenance prototype that detects abnormal gearbox sensor patterns locally using unsupervised machine learning and provides interpretable anomaly analysis.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-EC6A00?logo=xgboost\&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2)](https://shap.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite\&logoColor=white)](https://www.sqlite.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git\&logoColor=white)](https://git-scm.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

## 📌 Overview

**Edge-AutoGuard** is a student-level prototype for detecting **abnormal patterns in gearbox sensor signals** using edge-based machine learning.

The system processes raw gearbox sensor signals, converts them into statistical features, and uses an **Isolation Forest** model to identify unusual signal patterns.

Instead of sending raw sensor data to a cloud service, inference is performed locally to simulate an **edge-computing environment**.

The system also provides:

* 📊 Signal analysis
* 🤖 Unsupervised anomaly detection
* 🔍 Explainable anomaly analysis
* ⚡ Local edge inference
* 🗄️ SQL-based prediction history
* 🌐 FastAPI backend
* 📈 Streamlit monitoring dashboard

---

# 🎯 Problem Statement

Modern mechanical systems continuously generate sensor data.

For a gearbox, signals such as vibration and rotational measurements can contain useful information about the operating condition of the system.

A challenge is that real-world sensor datasets may not always contain reliable labels indicating exactly when a mechanical fault occurred.

Therefore, instead of artificially creating fault labels, Edge-AutoGuard uses **unsupervised anomaly detection** to identify sensor patterns that differ significantly from the learned normal behavior.

### Goal

> Detect unusual gearbox sensor behavior early and provide an interpretable indication of which extracted signal features are associated with the anomaly.

---

# 💡 Why This Project?

The project was developed to explore the intersection of:

* 🤖 Artificial Intelligence
* 🚗 Automotive/Mechanical systems
* ⚡ Edge Computing
* 📊 Signal Processing
* 🔍 Explainable AI
* 🗄️ SQL/Data Management

Rather than building another generic chatbot or classification application, the project focuses on a real engineering problem involving **sensor data and predictive maintenance**.

---

# 🏗️ System Architecture

```text
                 PHM Gearbox Dataset
                         │
                         ▼
                ┌─────────────────┐
                │ Raw Sensor Data │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Signal Windowing│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Feature         │
                │ Extraction      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Isolation Forest│
                │ Anomaly Model   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Edge Inference  │
                │ Local Laptop    │
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Anomaly Analysis         FastAPI
              │                     │
              └──────────┬──────────┘
                         ▼
                ┌─────────────────┐
                │ SQLite Database │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Streamlit       │
                │ Dashboard       │
                └─────────────────┘
```

---

# 🔄 Complete Workflow
<img width="1567" height="743" alt="image" src="https://github.com/user-attachments/assets/6f3e0bcb-8ec9-41ba-a1f3-581130c73a23" />
<img width="1077" height="833" alt="image" src="https://github.com/user-attachments/assets/2c692e7c-9a01-4de5-9f20-7b5d5a449078" />
<img width="1547" height="673" alt="image" src="https://github.com/user-attachments/assets/b950647b-a680-40c6-b8a7-8d461c15a4f7" />
<img width="1807" height="802" alt="image" src="https://github.com/user-attachments/assets/29acc18c-09e0-466b-a6c7-046e0b1d5e49" />

## 1️⃣ Raw Sensor Data

The project uses the **PHM 2009 gearbox dataset** containing raw sensor measurements.

The downloaded dataset consists of multiple CSV runs.

The raw dataset contains approximately:

* **560 runs**
* **74 million+ signal rows**
* **3 signal channels per run**

The dataset does not provide a simple supervised fault label in the raw files.

Therefore, the project does **not** artificially create fault labels.

---

## 2️⃣ Signal Windowing

Raw signals are divided into smaller fixed-size windows.

```text
Run_1
│
├── Window 1
├── Window 2
├── Window 3
├── Window 4
└── ...
```

Each window represents a small segment of sensor behavior.

This makes it possible to analyze the signal locally instead of loading the entire dataset into memory.

---

## 3️⃣ Feature Extraction

Statistical features are extracted from each signal window.

### Time-domain features

* Mean
* Standard deviation
* Variance
* RMS
* Minimum
* Maximum
* Peak-to-peak
* Kurtosis
* Skewness

For example:

```text
Signal_1
   │
   ├── Mean
   ├── Standard Deviation
   ├── RMS
   ├── Kurtosis
   └── Skewness

Signal_2
   │
   ├── Mean
   ├── Standard Deviation
   ├── RMS
   ├── Kurtosis
   └── Skewness

Signal_3
   │
   ├── Mean
   ├── Standard Deviation
   ├── RMS
   ├── Kurtosis
   └── Skewness
```

The resulting feature vector is much smaller than the original raw signal.

---

# 🤖 4. Anomaly Detection

Because the raw dataset does not contain trustworthy fault labels, the project uses:

### Isolation Forest

Isolation Forest is an unsupervised machine-learning algorithm designed to identify observations that are unusual compared with the rest of the data.

```text
Feature Vector
      │
      ▼
Isolation Forest
      │
      ├──────────────┐
      ▼              ▼
   Normal        Anomalous
```

The model produces an **anomaly score**.

### Important

The anomaly score is **not a probability of failure**.

For example:

```text
Anomaly Score: 0.4104
Status: Normal
```

The exact score interpretation depends on the model implementation and thresholding.

---

# ⚡ 5. Edge Computing

The trained anomaly-detection model runs locally.

For the prototype:

> **The laptop simulates the edge device.**

```text
Sensor Data
     │
     ▼
Local Feature Extraction
     │
     ▼
Local ML Model
     │
     ▼
Anomaly Score
     │
     ▼
Normal / Anomalous
```

### Why Edge?

Local inference can provide:

* Low latency
* Reduced dependency on internet connectivity
* Local processing of sensor information
* Faster immediate analysis

No cloud service is required for inference.

---

# 🔍 6. Explainable Anomaly Analysis

The system analyzes the extracted features associated with anomalous windows.

Instead of simply showing:

> `Anomaly detected`

the system attempts to show which signal features differ from the learned/baseline behavior.

Example:

```text
Anomalous Window

RMS_Signal_2
Kurtosis_Signal_1
Std_Signal_3
PeakToPeak_Signal_2
```

This makes the anomaly more interpretable.

> **Note:** The project does not force SHAP into the pipeline if the chosen anomaly-detection implementation does not provide a technically meaningful SHAP explanation. Feature-deviation analysis can be used instead.

---

# 🌐 7. FastAPI Backend

FastAPI provides a lightweight local REST API.

### Main endpoints

```text
GET  /api/health

POST /api/predictions

GET  /api/predictions

GET  /api/predictions/{run_id}
```

The API handles:

* Prediction submission
* Prediction retrieval
* Run history
* Database communication
* Input validation

---

# 🗄️ 8. SQLite Database

Prediction history is stored locally using SQLite.

### Example schema

```text
predictions
│
├── id
├── run_id
├── window_id
├── anomaly_score
├── status
├── feature_data
├── explanation
├── edge_device_id
└── created_at
```

This allows historical analysis without requiring a cloud database.

---

# 📊 9. SQL Analytics

SQL is used to retrieve useful information such as:

### High-anomaly windows

```sql
SELECT *
FROM predictions
ORDER BY anomaly_score;
```

### Run history

```sql
SELECT *
FROM predictions
WHERE run_id = 'Run_1'
ORDER BY window_id;
```

### Count anomalies

```sql
SELECT status, COUNT(*)
FROM predictions
GROUP BY status;
```

### Run-wise analysis

```sql
SELECT
    run_id,
    status,
    COUNT(*) AS total
FROM predictions
GROUP BY run_id, status;
```

---

# 📈 10. Streamlit Dashboard

The Streamlit dashboard provides a visual interface for monitoring the anomaly-detection system.

### Dashboard sections

#### Overall Analytics

```text
Total Windows
Normal Windows
Anomalous Windows
Anomaly %
```

#### Run-wise Analysis

```text
Run       Windows    Anomalies
Run_1     1000       25
Run_2     1000       31
Run_3     1000       18
```

Actual values are generated from the processed dataset.

#### Selected Anomaly

```text
Run: Run_3
Window: 247

Status: ANOMALOUS

Anomaly Score: [actual value]
```

#### Signal Visualization

Separate plots are provided for:

* Signal 1
* Signal 2
* Signal 3

This avoids scale differences hiding smaller signals.

---

# 🛠️ Technology Stack

| Technology          | Purpose                         |
| ------------------- | ------------------------------- |
| 🐍 Python           | Core development                |
| 🐼 Pandas           | Data processing                 |
| 🔢 NumPy            | Numerical computation           |
| 📊 Scikit-learn     | Machine learning                |
| 🌲 Isolation Forest | Anomaly detection               |
| 🔍 SHAP             | Explainability where applicable |
| ⚡ FastAPI           | REST API                        |
| 🗄️ SQLite          | Local SQL database              |
| 🧮 SQLAlchemy       | Database ORM                    |
| 📈 Streamlit        | Dashboard                       |
| 🧪 Pytest           | Testing                         |
| 🐙 Git/GitHub       | Version control                 |
| 💻 VS Code          | Development environment         |
| 🤖 GitHub Copilot   | Development assistance          |

---

# 📁 Project Structure

```text
edge-autoguard/
│
├── data/
│   ├── extracted_features.csv
│   ├── anomaly_results.csv
│   └── edge_autoguard.db
│
├── models/
│   ├── anomaly_model.pkl
│   └── feature_columns.json
│
├── notebooks/
│   └── dataset_inspection.ipynb
│
├── src/
│   ├── inspect_dataset.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── explain.py
│   └── edge_simulator.py
│
├── api/
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   └── crud.py
│
├── dashboard/
│   └── app.py
│
├── sql/
│   └── queries.sql
│
├── tests/
│   └── ...
│
├── requirements.txt
├── .gitignore
├── .env.example
├── Dockerfile
└── README.md
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/edge-autoguard.git

cd edge-autoguard
```

## 2. Create virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Step 1 — Inspect Dataset

```bash
python src/inspect_dataset.py
```

## Step 2 — Extract Features

```bash
python src/feature_extraction.py
```

## Step 3 — Train Anomaly Model

```bash
python src/train.py
```

## Step 4 — Run Edge Simulator

```bash
python src/edge_simulator.py
```

## Step 5 — Start FastAPI

```bash
uvicorn api.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Step 6 — Start Streamlit

```bash
streamlit run dashboard/app.py
```

---

# 🔬 Example Workflow

```text
Run_1.csv
    │
    ▼
Signal Window
    │
    ▼
Feature Extraction
    │
    ▼
┌───────────────────────┐
│ Mean                  │
│ Standard Deviation    │
│ RMS                   │
│ Variance              │
│ Kurtosis              │
│ Skewness              │
└───────────┬───────────┘
            │
            ▼
    Isolation Forest
            │
       ┌────┴────┐
       ▼         ▼
    Normal    Anomalous
       │         │
       └────┬────┘
            ▼
       SQLite DB
            │
            ▼
       FastAPI API
            │
            ▼
   Streamlit Dashboard
```

---

# 🧪 Current Prototype Output

Example:

```text
Run: Run_1
Window: 0

Status: Normal

Anomaly Score:
0.4104
```

Example extracted features:

```text
mean_Signal_1
std_Signal_1
variance_Signal_1
rms_Signal_1
min_Signal_1
max_Signal_1
peak_to_peak_Signal_1
kurtosis_Signal_1
skewness_Signal_1

...
```

Actual results depend on the processed dataset and model configuration.

---

# 📊 Evaluation Strategy

Because the raw dataset does not provide ground-truth fault labels, conventional supervised metrics such as:

* Accuracy
* Precision
* Recall
* F1-score

are **not appropriate for validating real fault-detection accuracy**.

Instead, the prototype evaluates:

* Anomaly-score distribution
* Anomaly percentage
* Run-wise anomaly distribution
* Stability across runs
* Signal behavior of anomalous windows
* Feature deviations
* Visual inspection of detected windows

If reliable labeled data becomes available, supervised evaluation can be added in a future version.

---

# 🎯 Use Case

### Scenario

A gearbox continuously generates sensor signals.

The edge system receives a small window of sensor data:

```text
Sensor signals
      ↓
Local processing
      ↓
Feature extraction
      ↓
Anomaly detection
```

If the signal pattern differs significantly from the learned behavior:

```text
⚠️ ANOMALOUS SENSOR PATTERN
```

The system stores the result locally and displays it through the dashboard.

A technician or engineer can then investigate the corresponding signal window.

### Important

An anomaly does **not** automatically mean a mechanical failure.

It indicates that the sensor pattern is unusual compared with the model's learned behavior.

---

# ⭐ Key Features

* ⚡ Local edge inference
* 🤖 Unsupervised machine learning
* 📊 Raw signal processing
* 🧮 Statistical feature extraction
* 🔍 Explainable anomaly analysis
* 🗄️ SQLite + SQL history
* ⚡ FastAPI backend
* 📈 Interactive Streamlit dashboard
* 💾 Memory-efficient dataset processing
* 🔧 Run-level anomaly monitoring

---

# 💡 What Makes This Project Different?

The individual technologies used in this project are not novel.

The project's contribution is the integration of:

```text
Raw gearbox signals
        +
Window-based feature extraction
        +
Unsupervised anomaly detection
        +
Local edge inference
        +
Interpretable feature analysis
        +
SQL-based history
        +
Interactive monitoring
```

The project specifically explores how **unlabeled sensor data can be analyzed locally to identify abnormal operating patterns** without artificially creating fault labels.

---

# ⚠️ Limitations

This is a research/academic prototype.

### Current limitations

* No physical vehicle hardware
* Laptop simulates the edge device
* Raw dataset has no verified fault labels
* Anomalies cannot be directly interpreted as mechanical failures
* No safety certification
* No production ECU integration
* No real-time vehicle deployment
* Isolation Forest performance depends on feature quality and contamination settings

---

# 🔮 Future Improvements

If reliable labeled automotive data becomes available:

### 1. Supervised fault classification

```text
Sensor Data
     ↓
XGBoost
     ↓
Specific Fault Class
```

### 2. Real-time hardware

Deploy the model on:

* Raspberry Pi
* NVIDIA Jetson
* Automotive edge computer

### 3. Time-frequency analysis

Add:

* FFT
* Spectrograms
* Wavelet transforms

### 4. Lightweight edge optimization

Investigate:

* Model compression
* Quantization
* ONNX Runtime
* TensorRT

### 5. Fleet monitoring

Add centralized monitoring for multiple vehicles/components.

### 6. Real automotive telemetry

Integrate data from actual vehicle sensors/ECU systems.

---

# 🧠 Interview Explanation

### What is the project?

> Edge-AutoGuard is an explainable gearbox sensor anomaly-detection prototype. It processes raw gearbox signals locally, extracts statistical features, and uses Isolation Forest to identify unusual sensor patterns. The results are stored in SQLite and visualized through a Streamlit dashboard.

### Why anomaly detection?

> The raw dataset doesn't provide reliable fault labels, so instead of inventing labels, I used unsupervised anomaly detection to identify abnormal sensor behavior.

### Why edge computing?

> Running inference locally reduces dependency on network connectivity and allows the system to analyze sensor data with low latency.

### Did you use physical hardware?

> No. I simulated the edge device on my laptop because I didn't have access to physical automotive hardware.

### Is an anomaly a confirmed gearbox fault?

> No. It indicates an unusual sensor pattern. Without ground-truth labels, I cannot claim that it represents a specific mechanical failure.

### Why is that important?

> It prevents the system from making unsupported safety-critical claims.

---

# 📚 Dataset

The project uses the PHM Society gearbox dataset.

Official source:

https://phmsociety.org/public-data-sets/

The raw dataset contains gearbox sensor measurements used for research and predictive-maintenance experimentation.

---

# 📜 License

This project is intended for educational and research purposes.

The dataset remains subject to its original source's terms and conditions.

---

# 👨‍💻 Author

**Kamalesh B**

B.Tech — Artificial Intelligence & Data Science

---

## ⭐ Project Summary

```text
Edge-AutoGuard
│
├── 🤖 Unsupervised ML
├── ⚡ Edge Inference
├── 📊 Signal Processing
├── 🔍 Explainable AI
├── 🗄️ SQL Database
├── ⚡ FastAPI
└── 📈 Streamlit
```

> **Detect locally. Explain clearly. Investigate intelligently.**
