# 🧤 HYGEIA GLOVE – AI Health Monitoring System

> Real-time AI-powered patient health monitoring using an **ESP32 Smart Glove**.
> Measures **Body Temperature**, **SpO₂** and **Pulse Rate**, classifies health status
> as **Normal / Warning / Critical**, and streams a live 30-minute trend to a web dashboard.

---

## 📁 Project Structure

```
Hygeia ML/
├── data/
│   └── glove_readings.csv        ← Hygeia Glove sensor data (cleaned)
├── templates/
│   └── index.html                ← Live web dashboard (served by Flask)
├── HYGEIA_AI_FINAL_RESULTS/      ← Training plots & metric reports
├── photos of plots/              ← Visual reference plots
│
├── hygeia_local.py               ← Train the AI model (run first)
├── server.py                     ← Flask web server (run second)
├── smartglove.ino                ← ESP32 Arduino firmware
│
├── hygeia_ai_model_final.h5      ← Saved Keras model
├── hygeia_scaler.pkl             ← Saved RobustScaler (auto-created on train)
├── requirements.txt              ← Python dependencies
├── classification_report.txt     ← Model evaluation
└── final_metrics_summary.txt     ← Accuracy summary
```

---

## 🏗️ System Architecture

```
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
│  │  AI Model  →  Normal / Warning / Critical│             │
│  │  30-min rolling buffer (deque, max 360) │             │
│  │  Sample data thread (demo mode)         │             │
│  └─────────────────────────────────────────┘             │
│              ↓ http://localhost:5000                      │
│  ┌──────────────────────────────────┐                    │
│  │  Web Dashboard (index.html)      │                    │
│  │  Real-time vitals + trend charts │                    │
│  └──────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────┘
```

---

## 🤖 AI Model Architecture

| Layer | Config |
|-------|--------|
| Input | Vital signs: Temp, SpO₂, Pulse + 3 engineered features |
| Conv1D × 2 | 64 → 128 filters, BatchNorm + Dropout(0.3) |
| Bidirectional LSTM × 2 | 64 → 32 units |
| Dense | 128 → 64 → 3-class softmax |

**Accuracy: 90.91% · Mean Confidence: 93.5%**

---

## 🚀 Quick Start

### Step 1 – Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 – Train the AI Model

```bash
python hygeia_local.py
```
Reads `data/glove_readings.csv`, trains the model, saves:
- `hygeia_ai_model_final.h5`
- `hygeia_scaler.pkl`
- Training plots in `HYGEIA_AI_FINAL_RESULTS/`

### Step 3 – Run the Web Server

```bash
python server.py
```

Open **http://localhost:5000** in your browser.

> 🟡 **Demo mode** starts automatically with simulated sample data.
> The dashboard auto-switches to **ESP32 LIVE** mode when the glove connects.

### Step 4 – Flash the ESP32

1. Open `smartglove.ino` in Arduino IDE
2. Install required libraries (Library Manager):
   - `SparkFun MAX3010x Sensor Library`
   - `Adafruit MLX90614 Library`
   - `ArduinoJson` (v6.x)
3. Set your WiFi credentials and PC IP in the sketch:
   ```cpp
   const char* WIFI_SSID   = "YOUR_WIFI";
   const char* WIFI_PASS   = "YOUR_PASS";
   const char* SERVER_HOST = "192.168.x.x";  // run ipconfig on Windows
   ```
4. Select board: **ESP32 Dev Module** · Upload

---

## 🔌 ESP32 Hardware Wiring

| Sensor Pin | ESP32 Pin | Notes |
|-----------|-----------|-------|
| MAX30102 SDA | GPIO 21 | Shared I2C |
| MAX30102 SCL | GPIO 22 | Shared I2C |
| MAX30102 VCC | 3.3V | ⚠️ NOT 5V |
| MAX30102 GND | GND | |
| MLX90614 SDA | GPIO 21 | Same bus |
| MLX90614 SCL | GPIO 22 | Same bus |
| MLX90614 VCC | 3.3V | |
| MLX90614 GND | GND | |
| Status LED | GPIO 2 | Built-in on most boards |


---

## 🏥 Health Classification Rules

| Status | Criteria |
|--------|----------|
| ✅ Normal | SpO₂ ≥ 95%, Temp 36.0–38.5°C, Pulse 55–120 BPM |
| ⚠️ Warning | SpO₂ 90–94%, Temp slightly out of range |
| 🚨 Critical | SpO₂ < 90%, Temp > 40°C or < 35°C, Pulse > 150 or < 40 |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web dashboard |
| POST | `/api/reading` | Receive ESP32 data `{temp, spo2, pulse}` |
| GET | `/api/data` | All 30-min readings (JSON array) |
| GET | `/api/current` | Latest reading |
| GET | `/api/status` | Server status |
| POST | `/api/sample/start` | Start demo data |
| POST | `/api/sample/stop` | Stop demo data |
| POST | `/api/clear` | Clear buffer |

---

## 📦 Dependencies

- Python 3.8+, TensorFlow 2.13, scikit-learn 1.3, Flask 3.0
- pandas, numpy, matplotlib, seaborn, scipy, h5py, joblib, flask-cors

---

*KDU Sem 4 Project · HYGEIA Glove AI Health Monitoring*
