# HYGEIA AI – Patient Health Monitoring System

> **An AI-powered real-time patient health monitoring platform.**  
> Continuously analyses **Body Temperature**, **Blood Oxygen Saturation (SpO₂)**, and **Pulse Rate**
> using a time-series deep learning model, classifies risk level, and streams live trends to a
> clinical web dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Architecture](#3-system-architecture)
4. [Data Collection & Validation](#4-data-collection--validation)
5. [Machine Learning Pipeline](#5-machine-learning-pipeline)
6. [AI Model Architecture](#6-ai-model-architecture)
7. [Risk Classification Rules](#7-risk-classification-rules)
8. [Software Stack & Server](#8-software-stack--server)
9. [Web Dashboard](#9-web-dashboard)
10. [API Reference](#10-api-reference)
11. [Project File Structure](#11-project-file-structure)
12. [System Workflow (End-to-End)](#12-system-workflow-end-to-end)
13. [Testing & Validation](#13-testing--validation)
14. [Quick Start Guide](#14-quick-start-guide)
15. [Future Works](#15-future-works)

---

## 1. Project Overview

**HYGEIA AI** is a clinical-grade health monitoring system developed as part of the KDU Semester 4 project. The system ingests patient vital sign readings continuously, applies a trained deep learning model for risk-level prediction, and presents a real-time dashboard for caretakers and medical staff.

Unlike traditional single-point threshold monitors, HYGEIA AI uses a **sliding time-window** approach — analysing the *trend* over the last 25 seconds of readings, not just the current value. This allows the AI to detect gradual deterioration (e.g., slowly dropping SpO₂) well before thresholds are crossed.

### Core Objectives
- Provide **real-time, AI-driven** risk classification: Normal · Warning · Critical.
- Eliminate manual threshold monitoring by doctors/nurses.
- Display an accessible, high-contrast clinical dashboard readable from across a room.
- Log all session data for post-analysis and model improvement.

---

## 2. Key Features

| Feature | Detail |
|---|---|
| Real-Time Vital Monitoring | Temp, SpO₂, Pulse refreshed every 5 seconds |
| Time-Series AI Prediction | 5-step sliding window detects trend-based anomalies |
| 3-Level Risk Classification | Normal · Warning · Critical with confidence scores |
| Interactive Simulation | Switch between patient condition modes from the UI |
| 30-Min Historical Trend | Rolling chart buffer (360 readings) |
| Rule-Based Fallback | If AI model fails to load, clinical rule-engine takes over |
| CSV Session Logging | Every demo reading is logged to `data/demo_session_log.csv` |
| REST API | Clean endpoints for data ingestion and status |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA SOURCE LAYER                                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  data_streamer.py (patient_vitals.csv row-by-row)        │  │
│  │    OR                                                     │  │
│  │  Internal Demo Generator (_sample_loop in server.py)     │  │
│  │    OR                                                     │  │
│  │  Live External Device → POST /api/reading                │  │
│  └──────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────┼───────────────────────────────────┘
                              │ JSON: { temp, spo2, pulse }
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FLASK SERVER (server.py)                                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  RobustScaler  →  Feature Engineering  →  Sequence      │   │
│  │  (hygeia_scaler.pkl)                    (5 readings)    │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │  AI Model (hygeia_ai_model_final.h5)                    │   │
│  │  Conv1D + BiLSTM → 3-class Softmax                      │   │
│  │  Output: Normal | Warning | Critical  + Confidence %    │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │  30-min Rolling Buffer (deque, maxlen=360)               │   │
│  └────────────────────────┬────────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────────┘
                            │ http://localhost:5000
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  WEB DASHBOARD (templates/index.html)                          │
│  - Live vital value cards (large, high-contrast)               │
│  - AI Risk Status badge + confidence bar                       │
│  - 30-min trend chart (Chart.js)                               │
│  - Simulation Controller (Normal / Critical modes)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Collection & Validation

### 4.1 Data Source
The training dataset (`data/patient_vitals.csv`) consists of patient vital sign readings. Each row contains:

| Column | Type | Description |
|--------|------|-------------|
| `Temp` | float | Body temperature in °C |
| `SpO2` | int | Blood oxygen saturation (%) |
| `Pulse` | int | Pulse rate in BPM |
| `Status` | string | Clinical label: Normal / Warning / Critical |

### 4.2 Data Validation (in `hygeia_local.py`)

Raw CSV data goes through a strict validation pipeline before it reaches the model:

```
Raw CSV
  │
  ▼
[1] Coerce non-numeric → NaN
  │
  ▼
[2] Range Filter (hard physiological bounds):
     - Temp:  30°C  < T  < 45°C
     - SpO2:   0%   < S  ≤ 100%
     - Pulse:  0    < P  < 200
  │
  ▼
[3] Drop remaining NaN rows
  │
  ▼
[4] Reset index
  │
  ▼
  Clean DataFrame → Feature Engineering → Model Training
```

### 4.3 Feature Engineering

Three base features are expanded to **six engineered features** to amplify anomaly signals:

| Feature | Formula | Purpose |
|---|---|---|
| `Temp` | raw | Body temperature |
| `SpO2` | raw | Blood oxygen |
| `Pulse` | raw | Pulse rate |
| `Temp_SpO2_ratio` | `Temp / (SpO2 + ε)` | Amplifies combined oxygen/temp risk |
| `Pulse_SpO2_ratio` | `Pulse / (SpO2 + ε)` | Amplifies stress on oxygen |
| `Health_Index` | `SpO2×0.5 - |Temp-37|×10 - |Pulse-75|×0.1` | Composite clinical score |

### 4.4 Normalisation

All six features are scaled using **RobustScaler** (scikit-learn):
- Robust to outliers (uses median and IQR instead of mean/std).
- Fitted **once** on training data and persisted as `hygeia_scaler.pkl`.
- Applied identically at inference time.

---

## 5. Machine Learning Pipeline

```
patient_vitals.csv
       │
       ▼
  Data Validation (range filter, NaN removal)
       │
       ▼
  Feature Engineering (6 features per reading)
       │
       ▼
  RobustScaler fit + transform
       │
       ▼
  Sequence Creation (sliding window, SEQ_LEN=5)
  Shape: (N-SEQ_LEN+1, 5, 6)
       │
       ▼
  ┌──────────────────────────────┐
  │  80% Train / 20% Test split  │
  └──────────────────────────────┘
       │                    │
       ▼                    ▼
  Model Training        Evaluation
  (150 epochs max)      Accuracy / Confidence /
  EarlyStopping         Confusion Matrix /
  ReduceLROnPlateau     Classification Report
       │
       ▼
  hygeia_ai_model_final.h5
  hygeia_scaler.pkl
```

### 5.1 Sequence (Sliding Window) Creation

Instead of classifying a single reading in isolation, every sample feeds in the last **5 consecutive readings** (25 seconds of history). This is created by:

```python
for i in range(len(X) - SEQ_LEN + 1):
    X_seq.append(X[i : i + SEQ_LEN])   # shape (5, 6)
    y_seq.append(y[i + SEQ_LEN - 1])   # label of the final reading
```

This forces the model to learn **temporal patterns**, not just snapshot values.

---

## 6. AI Model Architecture

```
Input: (batch, 5, 6)   ← 5 timesteps × 6 features
       │
       ▼
Conv1D(64, kernel=2, relu)  +  BatchNorm  +  Dropout(0.2)
       │
       ▼
Conv1D(128, kernel=2, relu) +  BatchNorm  +  Dropout(0.2)
       │
       ▼
Bidirectional LSTM(64 units)  +  Dropout(0.3)
       │
       ▼
Bidirectional LSTM(32 units)  +  Dropout(0.3)
       │
       ▼
Dense(64, relu)  +  Dropout(0.3)
       │
       ▼
Dense(3, softmax)
       │
       ▼
Output: [P(Normal), P(Warning), P(Critical)]
```

**Why this architecture?**
- **Conv1D layers** act as local pattern detectors — they learn what a "fast drop in SpO₂ over 3 steps" looks like.
- **Bidirectional LSTM layers** provide both forward and backward temporal context, capturing complex time dependencies.
- **BatchNorm** stabilises training and allows higher learning rates.
- **Softmax output** gives calibrated probability across all three classes.

### Training Callbacks

| Callback | Configuration | Purpose |
|---|---|---|
| EarlyStopping | patience=20, restore_best_weights | Stops before overfitting |
| ReduceLROnPlateau | factor=0.5, patience=10 | Reduces LR on val plateau |
| ModelCheckpoint | saves best val_accuracy | Keeps the best checkpoint |

### Performance

| Metric | Value |
|---|---|
| Overall Accuracy | 100% (on test split) |
| Mean Confidence | ~99.97% |
| Architecture | Conv1D × 2 + BiLSTM × 2 |
| Sequence Length | 5 readings (25 seconds) |

---

## 7. Risk Classification Rules

Used as the **rule-based fallback** (when the AI model is unavailable) and for initial dataset labelling:

```
                      ┌─────────────────────────────────────────┐
                      │           INCOMING VITALS               │
                      │   temp, spo2, pulse                     │
                      └──────────────────┬──────────────────────┘
                                         │
                   ┌─────────────────────▼────────────────────────┐
                   │  CRITICAL CHECK                              │
                   │  spo2 < 90  OR  temp > 40°C  OR temp < 35°C │
                   │  OR  pulse > 150  OR  pulse < 40            │
                   └──────────────────┬──────────────────────────-┘
                         YES ◄────────┘         │ NO
                          │                     ▼
                   [CRITICAL 🔴]       ┌──────────────────────┐
                   confidence=0.95     │  WARNING CHECK        │
                                       │  spo2 < 95  OR        │
                                       │  temp > 38.5 OR <36°C │
                                       │  OR pulse > 120/<55   │
                                       └────────────┬──────────┘
                                          YES ◄─────┘    │ NO
                                           │              ▼
                                   [WARNING 🟡]     [NORMAL 🟢]
                                   conf=0.85        conf=0.92
```

---

## 8. Software Stack & Server

| Component | Technology |
|---|---|
| Backend Server | Python 3.8 + Flask |
| AI Framework | TensorFlow 2.13 / Keras |
| Scaler | scikit-learn RobustScaler |
| Data Handling | pandas, numpy |
| CORS | flask-cors |
| Model Persistence | h5py (Keras), joblib (scaler) |
| Frontend | Vanilla HTML/CSS/JS, Chart.js |

### Key Design Decisions

**Stateful Sequence Buffer**: The server maintains a `deque(maxlen=5)` of scaled feature vectors (`recent_features`). Every incoming reading appends to this deque and the full sequence `(1, 5, 6)` is immediately fed to the model.

**Thread Safety**: All writes to `readings_buffer` and `recent_features` are guarded by `threading.Lock()`.

**Auto-Demo Mode**: The server auto-starts the internal demo generator on launch. Once a live external reading arrives, demo mode stops automatically.

**30-Minute Pre-population**: On startup, `pre_populate_history()` fills the chart buffer with 360 synthetically generated historical readings so the dashboard is immediately useful.

---

## 9. Web Dashboard

The dashboard is a single-page application (`templates/index.html`) served by Flask.

### Components

| Panel | Description |
|---|---|
| Header Bar | System name, live clock, connection status badge |
| Vital Cards | Large-font temperature, SpO₂, and pulse displays with colour coding |
| AI Status Card | Current risk classification (Normal/Warning/Critical) + confidence progress bar |
| 30-Min Chart | Line chart of all three vitals over the last 30 minutes |
| Simulation Controller | Four buttons to switch patient condition scenario in real time |
| Footer | Classifier type indicator (AI vs Rule-based) |

### Simulation Modes (visible in Demo Mode)

| Button | Scenario | Target Values |
|---|---|---|
| Normal Patient | Healthy baseline | Temp ~36.8°C, SpO₂ ~98%, Pulse ~75 BPM |
| Drop SpO₂ | Hypoxaemia simulation | SpO₂ drops to ~85% |
| Spike Temp | High fever simulation | Temp rises to ~41.5°C |
| High Pulse | Tachycardia simulation | Pulse spikes to ~165 BPM |

Each press of a mode button sends `POST /api/simulation/mode` to the server, which immediately steers the Gaussian random walk toward the critical target values.

---

## 10. API Reference

| Method | Endpoint | Request Body | Response | Description |
|--------|----------|-------------|----------|-------------|
| `GET` | `/` | — | HTML | Serves the web dashboard |
| `POST` | `/api/reading` | `{"temp", "spo2", "pulse"}` | `{status, classification, confidence}` | Ingest a live vital reading |
| `GET` | `/api/data` | — | JSON array (up to 360) | All buffered readings |
| `GET` | `/api/current` | — | JSON object | Most recent reading only |
| `GET` | `/api/status` | — | JSON object | Server + model + simulation status |
| `POST` | `/api/sample/start` | — | `{status}` | Start demo data generator |
| `POST` | `/api/sample/stop` | — | `{status}` | Stop demo data generator |
| `POST` | `/api/simulation/mode` | `{"mode": "normal|critical_spo2|critical_temp|critical_pulse"}` | `{status, mode}` | Change simulation scenario |
| `POST` | `/api/clear` | — | `{status}` | Clear all buffers |

### Example: Posting a Reading
```bash
curl -X POST http://localhost:5000/api/reading \
  -H "Content-Type: application/json" \
  -d '{"temp": 38.2, "spo2": 93, "pulse": 118}'
```

Response:
```json
{
  "status": "ok",
  "classification": "Warning",
  "confidence": 0.8912
}
```

---

## 11. Project File Structure

```
Hygeia ML/
│
├── data/
│   ├── patient_vitals.csv         ← Collected patient vital readings (training data)
│   └── demo_session_log.csv       ← Auto-logged demo/simulation session output
│
├── templates/
│   └── index.html                 ← Web dashboard (Flask-served SPA)
│
├── HYGEIA_AI_FINAL_RESULTS/       ← Training plots (loss curves, confusion matrix)
├── photos of plots/               ← Visual reference plots
│
├── hygeia_local.py                ← Model training script (run once to produce .h5)
├── server.py                      ← Flask web server (main runtime)
├── data_streamer.py               ← Streams patient_vitals.csv to server API
├── start_demo.bat                 ← One-click Windows launcher (server + streamer + browser)
├── sensor_firmware.ino            ← Reference ESP32 firmware (hardware integration)
│
├── hygeia_ai_model_final.h5       ← Trained Keras model (output of hygeia_local.py)
├── hygeia_scaler.pkl              ← Fitted RobustScaler (output of hygeia_local.py)
├── requirements.txt               ← Python package dependencies
├── classification_report.txt      ← Per-class precision/recall report
└── final_metrics_summary.txt      ← Summary accuracy metrics
```

---

## 12. System Workflow (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 – DATA COLLECTION                                             │
│  Patient vitals recorded and saved to data/patient_vitals.csv          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 – DATA VALIDATION & PREPROCESSING (hygeia_local.py)           │
│  1. Load CSV                                                            │
│  2. Range validation (hard physiological bounds)                        │
│  3. NaN removal                                                         │
│  4. Feature engineering (3 → 6 features)                               │
│  5. RobustScaler fit and transform                                      │
│  6. Sliding window sequence creation (SEQ_LEN=5)                       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3 – MODEL TRAINING (hygeia_local.py)                            │
│  Conv1D × 2 + BiLSTM × 2 + Dense → 3-class Softmax                    │
│  Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint          │
│  Outputs: hygeia_ai_model_final.h5 + hygeia_scaler.pkl                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4 – SERVER RUNTIME (server.py)                                  │
│  1. Load .h5 model + .pkl scaler                                        │
│  2. Pre-populate 30-min history buffer                                  │
│  3. Start demo generator (or receive live readings via /api/reading)   │
│  4. Each reading → scale → append to deque → model.predict()          │
│  5. Return: status (Normal/Warning/Critical) + confidence              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 5 – DASHBOARD (http://localhost:5000)                           │
│  JS polls /api/data every 3 seconds                                    │
│  Renders: vital cards + AI risk badge + 30-min chart                   │
│  User can switch patient scenarios via Simulation Controller           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Testing & Validation

### 13.1 Model Testing
After training, the model is evaluated on a held-out 20% test set:
- **Classification Report**: Per-class precision, recall, F1-score (saved to `classification_report.txt`).
- **Confusion Matrix**: Visualised and saved to `HYGEIA_AI_FINAL_RESULTS/`.
- **Mean Confidence**: Average softmax probability of the predicted class.

### 13.2 Simulation-Based Testing (Functional)
The `start_demo.bat` launcher enables end-to-end functional testing without hardware:
1. `server.py` starts and loads the AI model.
2. `data_streamer.py` feeds rows from `patient_vitals.csv` to `/api/reading`.
3. Dashboard is opened and the team verifies:
   - Vital values update correctly.
   - AI risk status changes as expected.
   - Trend chart scrolls forward in real time.

### 13.3 Simulation Mode Testing (Critical Scenarios)
Using the **Simulation Controller** in the dashboard:
- **Drop SpO₂**: Verify status transitions from Normal → Warning → Critical as SpO₂ approaches 85%.
- **Spike Temp**: Verify Critical trigger when Temp exceeds 40°C.
- **High Pulse**: Verify Critical trigger when Pulse exceeds 150 BPM.
- **Return to Normal**: Verify smooth recovery back to Normal status.

### 13.4 Fallback Rule Engine Testing
- Temporarily rename `hygeia_ai_model_final.h5` to test rule-based classification.
- Verify dashboard still correctly classifies critical synthetic inputs.

---

## 14. Quick Start Guide

### Prerequisites
- Python 3.8 or later
- Windows OS (for `.bat` launcher)
- Internet connection for first-time package install

### Option A – One-Click Demo (Recommended)
Simply double-click **`start_demo.bat`**.

This automatically:
1. Launches the Flask AI server.
2. Waits 6 seconds for it to initialise.
3. Opens the dashboard in your default browser.
4. Starts streaming data from `patient_vitals.csv`.

### Option B – Manual Setup

**Step 1 – Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 – Train the AI model** *(skip if `hygeia_ai_model_final.h5` already exists)*
```bash
python hygeia_local.py
```

**Step 3 – Start the server**
```bash
python server.py
```

**Step 4 – Open the dashboard**

Navigate to **http://localhost:5000** in any browser.

**Step 5 (Optional) – Stream CSV data as a live feed**
```bash
python data_streamer.py
```

---

## 15. Future Works

The following features are planned for upcoming development iterations:

- **Automated Critical Alert Notification System**  
  Implement an SMS, Email, or Push Notification mechanism. When a patient's status transitions to **Critical**, an automated alert is immediately dispatched to registered doctors, nurses, and family members — regardless of whether anyone is actively watching the dashboard.

- **Cloud Deployment**  
  Migrate the local Flask backend to AWS EC2 or Google Cloud Run to enable ubiquitous access to patient monitoring data from any location.

- **Patient Session Management**  
  Add a multi-patient interface where different patients can be individually tracked, with their own historical data, notes, and trend graphs.

- **PDF/CSV Report Export**  
  Allow medical staff to export complete daily or weekly vital sign logs as formatted PDF reports for clinical records and audits.

- **Mobile Application**  
  Develop a dedicated Flutter or React Native mobile app so family members and on-call medical staff can check patient status from their phones at any time.

- **Anomaly Detection Pre-Training**  
  Augment the AI pipeline with an unsupervised anomaly detection layer (Autoencoder or Isolation Forest) to flag rare critical patterns that may fall outside the supervised training distribution.

---

*KDU Semester 4 Project · HYGEIA AI Patient Health Monitoring System*
