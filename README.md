# 🧤 HYGEIA GLOVE – AI Health Monitoring System

> Real-time AI-powered patient health monitoring using an **ESP32 Smart Glove**.
> Measures **Body Temperature**, **SpO₂**, and **Pulse Rate**, classifies health status 
> dynamically using time-series deep learning, and streams a live trend to a web dashboard.

---

## 📖 Table of Contents

1. [🌟 Project Overview](#1--project-overview)
2. [✨ Key Features](#2--key-features)
3. [🏗️ System Architecture](#3-️-system-architecture)
4. [🔌 Hardware Details (ESP32)](#4--hardware-details-esp32)
5. [🧠 Machine Learning Methodology](#5--machine-learning-methodology)
6. [🏥 Health Classification Rules](#6--health-classification-rules)
7. [💻 Software Details (Web Server)](#7--software-details-web-server)
8. [🌐 API Endpoints](#8--api-endpoints)
9. [📁 Project Structure](#9--project-structure)
10. [🚀 Quick Start Guide](#10--quick-start-guide)
11. [🔮 Future Works](#11--future-works)

---

## 1. 🌟 Project Overview

The **HYGEIA Glove** is an advanced wearable device designed to provide non-invasive, continuous health monitoring. By collecting vital signs in real-time, the system bridges the gap between patient care and remote health tracking. It utilizes a robust Artificial Intelligence model deployed via a local Flask server to instantly categorize the patient's condition as **Normal**, **Warning**, or **Critical**, providing an intuitive dashboard for caretakers to monitor 30-minute historical trends.

---

## 2. ✨ Key Features

- **Continuous Vital Monitoring**: Real-time tracking of Body Temperature, Blood Oxygen Saturation (SpO₂), and Pulse Rate.
- **Time-Series Deep Learning Prediction**: Analyzes sliding windows of historical data to detect sudden drops or spikes in vitals.
- **Sleek Web Dashboard**: Features smooth transitions, interactive live charts (via Chart.js), and a visually striking dark mode interface.
- **Rule-Based Fallback Engine**: If the AI model fails to load, the system guarantees safety by falling back to robust clinical thresholds.
- **Demo Mode**: Built-in simulated patient data generation for testing the dashboard without connecting the hardware.

---

## 3. 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────┐
│  HYGEIA SMART GLOVE (ESP32)                              │
│  ┌─────────────┐  ┌──────────────────┐                  │
│  │ MLX90614    │  │  MAX30102        │                  │
│  │ Temperature │  │  SpO2 + BPM      │                  │
│  └──────┬──────┘  └────────┬─────────┘                  │
│         └──────────────────┘                             │
│              ↓ I2C (SDA=21, SCL=22)                      │
│         [ ESP32 ]  →  HTTP POST /api/reading             │
└──────────────────────────────┬───────────────────────────┘
                               │ JSON: {temp, spo2, pulse}
                               ↓ WiFi
┌──────────────────────────────────────────────────────────┐
│  PC  –  server.py (Flask)                                │
│  ┌─────────────────────────────────────────┐             │
│  │  AI Model (Sequence window=5)           │             │
│  │  Classification: Normal/Warning/Critical│             │
│  │  30-min rolling buffer (deque, max 360) │             │
│  └─────────────────────────────────────────┘             │
│              ↓ http://localhost:5000                     │
│  ┌──────────────────────────────────┐                    │
│  │  Web Dashboard (index.html)      │                    │
│  │  Real-time vitals + trend charts │                    │
│  └──────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────┘
```

---

## 4. 🔌 Hardware Details (ESP32)

The system relies on an **ESP32 Dev Module** acting as the main microcontroller. 

### Core Sensors
1. **MAX30102**: An integrated pulse oximetry and heart-rate monitor module.
2. **MLX90614**: An infrared thermometer for non-contact temperature measurements.

### Wiring Guide
| Sensor Pin | ESP32 Pin | Notes |
|-----------|-----------|-------|
| MAX30102 SDA | GPIO 21 | Shared I2C Bus |
| MAX30102 SCL | GPIO 22 | Shared I2C Bus |
| MAX30102 VCC | 3.3V | ⚠️ STRICTLY 3.3V, NOT 5V |
| MAX30102 GND | GND | |
| MLX90614 SDA | GPIO 21 | Shared I2C Bus |
| MLX90614 SCL | GPIO 22 | Shared I2C Bus |
| MLX90614 VCC | 3.3V | |
| MLX90614 GND | GND | |
| Status LED | GPIO 2 | Built-in on most boards |

<p><strong>Note:</strong> The MAX30102 and MLX90614 share the same I²C bus using GPIO 21 (SDA) and GPIO 22 (SCL), allowing both sensors to communicate with the ESP32 through the same I²C connection.</p>
---

## 5. 🧠 Machine Learning Methodology

The AI acts as an early-warning system. Instead of merely checking static thresholds, it uses a time-series methodology to observe the *trend* of the vitals.

- **Data Preprocessing**: Tabular data is transformed into sequences (`SEQ_LEN=5`), equivalent to 25 seconds of medical history.
- **Feature Engineering**: Calculates ratios like `Temp/SpO2` and a composite `Health_Index` to amplify anomaly signals.
- **Network Architecture**: 
  - **Inputs**: Shape `(SEQ_LEN, 6)`
  - **Spatial Extraction**: 2 × Conv1D layers (64 & 128 filters) with Batch Normalization.
  - **Temporal Extraction**: 2 × Bidirectional LSTM layers (64 & 32 units).
  - **Classification**: Dense layers culminating in a 3-class Softmax layer.

**Performance Metrics**: Accuracy: 90.91% · Mean Confidence: 93.5%

---

## 6. 🏥 Health Classification Rules

When acting as a fallback (or labelling the original dataset), these clinical rules define the statuses:

| Status | Clinical Criteria | Action Required |
|--------|-------------------|-----------------|
| ✅ **Normal** | SpO₂ ≥ 95%, Temp 36.0–38.5°C, Pulse 55–120 BPM | None, continue monitoring. |
| ⚠️ **Warning** | SpO₂ 90–94%, Temp slightly out of range | Close observation needed. |
| 🚨 **Critical** | SpO₂ < 90%, Temp > 40°C or < 35°C, Pulse > 150 or < 40 | Immediate medical intervention! |

---

## 7. 💻 Software Details (Web Server)

The backend is driven by **Flask (Python)**, managing:
- Multi-threaded sensor simulation for demo environments.
- Real-time `deque` buffering (maintaining exactly the latest 360 readings, i.e., 30 minutes at 5-second intervals).
- Integration with TensorFlow Keras models and Scikit-Learn RobustScaler for real-time inference.

---

## 8. 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the HTML Web Dashboard |
| POST | `/api/reading` | Ingests ESP32 data `{"temp", "spo2", "pulse"}` |
| GET | `/api/data` | Returns all 30-min rolling buffer readings (JSON array) |
| GET | `/api/current` | Returns only the latest reading |
| GET | `/api/status` | Provides server health and classifier status |
| POST | `/api/sample/start`| Activates the multi-threaded demo data generator |
| POST | `/api/sample/stop` | Halts demo data generation |
| POST | `/api/clear` | Flushes all historical readings and AI sequences |

---

## 9. 📁 Project Structure

```text
Hygeia ML/
├── data/
│   └── glove_readings.csv        ← Hygeia Glove sensor data (cleaned)
├── templates/
│   └── index.html                ← Live web dashboard (served by Flask)
├── HYGEIA_AI_FINAL_RESULTS/      ← Training plots & metric reports
├── photos of plots/              ← Visual reference plots
│
├── hygeia_local.py               ← Train the AI model script
├── server.py                     ← Flask web server script
├── smartglove.ino                ← ESP32 Arduino firmware
│
├── hygeia_ai_model_final.h5      ← Saved Keras AI model
├── hygeia_scaler.pkl             ← Saved RobustScaler (auto-created on train)
├── requirements.txt              ← Python dependencies
├── classification_report.txt     ← Model evaluation report
└── final_metrics_summary.txt     ← Accuracy summary report
```

---

## 10. 🚀 Quick Start Guide

### Step 1 – Install Dependencies
Ensure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt
```

### Step 2 – Train the AI Model
```bash
python hygeia_local.py
```
This reads `data/glove_readings.csv`, creates time-series sequences, and trains the Deep Learning model, outputting the `.h5` file and `.pkl` scaler.

### Step 3 – Run the Web Server
```bash
python server.py
```
Open **http://localhost:5000** in your browser.
> 🟡 **Demo mode** starts automatically with simulated physiological data mimicking human behavior.
> The dashboard auto-switches to **ESP32 LIVE** mode the moment a real glove connects via Wi-Fi.

### Step 4 – Flash the ESP32 (Hardware Deployment)
1. Open `smartglove.ino` in Arduino IDE.
2. Install required libraries (via Library Manager):
   - `SparkFun MAX3010x Sensor Library`
   - `Adafruit MLX90614 Library`
   - `ArduinoJson` (v6.x)
3. Modify the WiFi credentials and point `SERVER_HOST` to your PC's local IP address:
   ```cpp
   const char* WIFI_SSID   = "YOUR_WIFI";
   const char* WIFI_PASS   = "YOUR_PASS";
   const char* SERVER_HOST = "192.168.x.x";  // Check with `ipconfig` or `ifconfig`
   ```
4. Select board: **ESP32 Dev Module**, then compile and **Upload**.

---

## 11. 🔮 Future Works

As the HYGEIA Glove continues to evolve, several major features are scheduled for upcoming releases:

- **Notification Alert Message System**: Implement an automated SMS, Email, or Push Notification mechanism. This ensures doctors, nurses, and caretakers are immediately and automatically alerted if a patient's vitals cross into a **Critical** situation, regardless of whether they are actively watching the dashboard.
- **Cloud Integration**: Migrate the local Flask backend to AWS or Google Cloud to allow ubiquitous access to patient data from any hospital ward in the world.
- **Patient History Export**: Add PDF/CSV generation features for doctors to download daily or weekly vital sign logs for permanent medical records.
- **Mobile Application**: Build a dedicated Flutter or React Native mobile app for family members to check on their loved ones seamlessly.

---

*KDU Sem 4 Project · HYGEIA Glove AI Health Monitoring*
