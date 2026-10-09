"""
HYGEIA AI – Patient Health Monitoring Server
=============================================
Generates AI-classified patient vital sign readings,
serves the web dashboard, and streams 30-minute trend history.

Run:      python server.py
Dashboard: http://localhost:5000
"""

import os
import random
import threading
import time
import json
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

# -- Optional ML imports -------------------------------------------------------
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not available - using rule-based classification")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from sklearn.preprocessing import RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# -- Flask app -----------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Shared state --------------------------------------------------------------
# 30 minutes @ 5-second intervals = 360 readings max
readings_buffer: deque = deque(maxlen=360)
recent_features: deque = deque(maxlen=5)   # SEQ_LEN = 5
buffer_lock = threading.Lock()
sample_running   = False
sample_thread    = None
live_connected   = False          # True when a live external source posts data
last_live_time   = None
simulation_mode  = "normal"

model  = None
scaler = None

STATUS_NAMES  = {0: "Normal", 1: "Warning", 2: "Critical"}
STATUS_COLORS = {0: "#00ff88", 1: "#ffcc00", 2: "#ff4444"}

# ==============================================================================
#  MODEL & SCALER LOADING
# ==============================================================================

def load_model_and_scaler():
    global model, scaler

    model_path = os.path.join(BASE_DIR, "hygeia_ai_model_final.h5")
    if TF_AVAILABLE and os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"AI model loaded  -> {model_path}")
        except Exception as e:
            print(f"Model load failed: {e}")

    scaler_path = os.path.join(BASE_DIR, "hygeia_scaler.pkl")
    if JOBLIB_AVAILABLE and os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"Scaler loaded    -> {scaler_path}")
            return
        except Exception as e:
            print(f"Scaler load failed: {e}")

    # Fit scaler from training CSV as fallback
    if SKLEARN_AVAILABLE:
        csv_path = os.path.join(BASE_DIR, "data", "patient_vitals.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path).dropna()
            df["Temp_SpO2_ratio"]  = df["Temp"]  / (df["SpO2"]  + 1e-6)
            df["Pulse_SpO2_ratio"] = df["Pulse"] / (df["SpO2"]  + 1e-6)
            df["Health_Index"]     = (
                df["SpO2"] * 0.5
                - abs(df["Temp"]  - 37.0) * 10
                - abs(df["Pulse"] - 75.0) * 0.1
            )
            features = ["Temp", "SpO2", "Pulse",
                        "Temp_SpO2_ratio", "Pulse_SpO2_ratio", "Health_Index"]
            scaler = RobustScaler()
            scaler.fit(df[features].values)
            print("Scaler fitted from patient_vitals.csv")

# ==============================================================================
#  CLASSIFICATION
# ==============================================================================

def rule_based_classify(temp: float, spo2: float, pulse: float):
    """Deterministic rule-based fallback classifier."""
    if spo2 < 90 or temp > 40.0 or temp < 35.0 or pulse > 150 or pulse < 40:
        return 2, 0.95
    if spo2 < 95 or temp > 38.5 or temp < 36.0 or pulse > 120 or pulse < 55:
        return 1, 0.85
    return 0, 0.92


def classify(temp: float, spo2: float, pulse: float):
    """Returns (status_idx, confidence, status_name)."""
    global recent_features
    if model is not None and scaler is not None:
        try:
            feats = np.array([[
                temp, spo2, pulse,
                temp  / (spo2  + 1e-6),
                pulse / (spo2  + 1e-6),
                spo2 * 0.5 - abs(temp - 37.0) * 10 - abs(pulse - 75.0) * 0.1
            ]])
            scaled_feat = scaler.transform(feats)[0]

            with buffer_lock:
                if len(recent_features) == 0:
                    for _ in range(5):
                        recent_features.append(scaled_feat)
                else:
                    recent_features.append(scaled_feat)
                scaled_seq = np.array(recent_features).reshape(1, 5, 6)

            probs = model.predict(scaled_seq, verbose=0)[0]
            idx   = int(np.argmax(probs))
            return idx, float(np.max(probs)), STATUS_NAMES[idx]
        except Exception as e:
            print(f"Inference error: {e}")

    idx, conf = rule_based_classify(temp, spo2, pulse)
    return idx, conf, STATUS_NAMES[idx]


def make_reading(temp, spo2, pulse, source="demo"):
    idx, conf, name = classify(temp, spo2, pulse)
    return {
        "timestamp":  datetime.now().isoformat(),
        "temp":       round(float(temp),  1),
        "spo2":       round(float(spo2),  0),
        "pulse":      round(float(pulse), 0),
        "status":     name,
        "status_idx": idx,
        "confidence": round(conf, 4),
        "color":      STATUS_COLORS[idx],
        "source":     source,
    }


def _sample_loop():
    global sample_running, simulation_mode
    temp  = random.uniform(36.5, 37.2)
    spo2  = random.uniform(96.0, 99.0)
    pulse = random.uniform(68.0, 88.0)

    csv_file = os.path.join(BASE_DIR, "data", "demo_session_log.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write("Timestamp,Temp,SpO2,Pulse,Mode\n")

    while sample_running:
        if simulation_mode == "normal":
            target_temp, target_spo2, target_pulse = 36.8, 98.0, 75.0
        elif simulation_mode == "critical_spo2":
            target_temp, target_spo2, target_pulse = 37.0, 85.0, 110.0
        elif simulation_mode == "critical_temp":
            target_temp, target_spo2, target_pulse = 41.5, 96.0, 125.0
        elif simulation_mode == "critical_pulse":
            target_temp, target_spo2, target_pulse = 36.9, 97.0, 165.0
        else:
            target_temp, target_spo2, target_pulse = 36.8, 98.0, 75.0

        # Steer 20% towards target each tick + Gaussian noise
        temp  = temp  + (target_temp  - temp)  * 0.2 + random.gauss(0, 0.05)
        spo2  = spo2  + (target_spo2  - spo2)  * 0.2 + random.gauss(0, 0.3)
        pulse = pulse + (target_pulse - pulse) * 0.2 + random.gauss(0, 1.5)

        temp  = max(30.0, min(45.0, temp))
        spo2  = max(50.0, min(100.0, spo2))
        pulse = max(30.0, min(220.0, pulse))

        t, s, p = round(temp, 1), round(spo2, 0), round(pulse, 0)
        reading = make_reading(t, s, p, source="demo")

        try:
            with open(csv_file, "a") as f:
                f.write(f"{reading['timestamp']},{t},{s},{p},{simulation_mode}\n")
        except Exception:
            pass

        with buffer_lock:
            readings_buffer.append(reading)

        time.sleep(5)


def pre_populate_history():
    """Fills the buffer with 30 minutes of simulated historical data."""
    temp  = 36.8
    spo2  = 98.0
    pulse = 78.0
    now   = datetime.now()
    history = []

    for i in range(360):
        offset = timedelta(seconds=(360 - i) * 5)
        ts = (now - offset).isoformat()

        temp  = max(35.5, min(40.5, temp  + random.gauss(0, 0.05)))
        spo2  = max(88.0, min(100.0, spo2  + random.gauss(0, 0.4)))
        pulse = max(45.0, min(155.0, pulse + random.gauss(0, 1.8)))

        t, s, p = round(temp, 1), round(spo2, 0), round(pulse, 0)
        idx, conf, name = classify(t, s, p)
        history.append({
            "timestamp":  ts,
            "temp":       t,
            "spo2":       s,
            "pulse":      p,
            "status":     name,
            "status_idx": idx,
            "confidence": round(conf, 4),
            "color":      STATUS_COLORS[idx],
            "source":     "history",
        })

    with buffer_lock:
        for r in history:
            readings_buffer.append(r)

    print(f"Pre-populated {len(history)} historical readings (30 min)")

# ==============================================================================
#  ROUTES
# ==============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reading", methods=["POST"])
def receive_reading():
    """External source posts: { "temp": 36.8, "spo2": 98, "pulse": 82 }"""
    global live_connected, last_live_time, sample_running

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    try:
        temp  = float(data["temp"])
        spo2  = float(data["spo2"])
        pulse = float(data["pulse"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Bad field: {e}"}), 400

    live_connected = True
    last_live_time = datetime.now()
    if sample_running:
        sample_running = False

    reading = make_reading(temp, spo2, pulse, source="live")
    with buffer_lock:
        readings_buffer.append(reading)

    return jsonify({
        "status": "ok",
        "classification": reading["status"],
        "confidence": reading["confidence"],
    })


@app.route("/api/data")
def get_data():
    """Return all buffered readings (up to 360)."""
    with buffer_lock:
        return jsonify(list(readings_buffer))


@app.route("/api/current")
def get_current():
    """Return the most recent reading."""
    with buffer_lock:
        if readings_buffer:
            return jsonify(readings_buffer[-1])
    return jsonify({})


@app.route("/api/status")
def get_status():
    alive = (
        live_connected
        and last_live_time is not None
        and (datetime.now() - last_live_time).total_seconds() < 15
    )
    return jsonify({
        "live_connected":  alive,
        "sample_running":  sample_running,
        "simulation_mode": simulation_mode,
        "readings_count":  len(readings_buffer),
        "model_loaded":    model is not None,
        "classifier":      "AI (TensorFlow)" if model is not None else "Rule-based",
    })


@app.route("/api/sample/start", methods=["POST"])
def start_sample():
    global sample_thread, sample_running
    if not sample_running:
        sample_running = True
        sample_thread  = threading.Thread(target=_sample_loop, daemon=True)
        sample_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/sample/stop", methods=["POST"])
def stop_sample():
    global sample_running
    sample_running = False
    return jsonify({"status": "stopped"})


@app.route("/api/simulation/mode", methods=["POST"])
def set_sim_mode():
    global simulation_mode
    data = request.get_json(silent=True)
    if data and "mode" in data:
        simulation_mode = data["mode"]
    return jsonify({"status": "ok", "mode": simulation_mode})


@app.route("/api/clear", methods=["POST"])
def clear_data():
    global recent_features
    with buffer_lock:
        readings_buffer.clear()
        recent_features.clear()
    return jsonify({"status": "cleared"})

# ==============================================================================
#  ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print()
    print("=" * 50)
    print("   HYGEIA AI - Patient Health Monitoring System")
    print("=" * 50)

    load_model_and_scaler()
    pre_populate_history()

    # Auto-start demo data (stops when a live external source connects)
    sample_running = True
    sample_thread  = threading.Thread(target=_sample_loop, daemon=True)
    sample_thread.start()
    print("Demo mode running - waiting for live data source")

    print()
    print(f"Dashboard : http://localhost:5000")
    print(f"API POST  : http://<your-pc-ip>:5000/api/reading")
    print("=" * 50)
    print()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
