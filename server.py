"""
HYGEIA AI – Patient Health Monitoring Server
=============================================
This is the main runtime for the HYGEIA AI system. It:

  1. Loads the trained AI model (models/hygeia_ai_model_final.h5) and scaler
     (models/hygeia_scaler.pkl) at startup.
  2. Pre-populates the dashboard with 30 minutes of simulated historical data.
  3. Auto-starts a demo data generator that steers vital readings toward
     realistic target values.
  4. Exposes a REST API for the web dashboard and external devices (e.g.,
     the ESP32-based HYGEIA Glove).
  5. Serves the web dashboard at http://localhost:5000.

Usage:
    python server.py

Dashboard:
    http://localhost:5000

External device integration (e.g., ESP32):
    POST http://<your-pc-ip>:5000/api/reading
    Body: { "temp": 36.8, "spo2": 98, "pulse": 82 }
"""

import os
import random
import threading
import time
import json
import argparse
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS


# ==============================================================================
#  OPTIONAL DEPENDENCY IMPORTS
#  The server degrades gracefully if TensorFlow is not installed.
# ==============================================================================

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not installed. Falling back to rule-based classification.")

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


# ==============================================================================
#  FLASK APPLICATION SETUP
# ==============================================================================

app = Flask(__name__, template_folder="templates")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
#  SHARED STATE
#  All mutable state is guarded by a threading lock to ensure thread safety
#  between the demo generator thread and the Flask request handlers.
# ==============================================================================

# Rolling 30-minute buffer: 5-second intervals × 360 = 30 minutes of readings
readings_buffer: deque = deque(maxlen=360)

# Keeps the last 5 scaled feature vectors for the time-series model input (SEQ_LEN=5)
recent_features: deque = deque(maxlen=5)

buffer_lock = threading.Lock()

# Demo generator state
sample_running  = False
sample_thread   = None
simulation_mode = "normal"   # Options: normal | critical_spo2 | critical_temp | critical_pulse

# Live device connection tracking
live_connected = False
last_live_time = None

# AI model and scaler (loaded at startup)
model  = None
scaler = None

# Classification constants
STATUS_NAMES  = {0: "Normal", 1: "Warning", 2: "Critical"}
STATUS_COLORS = {0: "#00ff88", 1: "#ffcc00", 2: "#ff4444"}


# ==============================================================================
#  MODEL & SCALER LOADING
# ==============================================================================

def load_model_and_scaler():
    """
    Load the trained Keras model and RobustScaler from the models/ directory.

    Falls back to fitting a fresh scaler from patient_vitals.csv if the
    saved scaler file is missing or corrupted.
    """
    global model, scaler

    # Load the Keras model
    model_path = os.path.join(BASE_DIR, "models", "hygeia_ai_model_final.h5")
    if TF_AVAILABLE and os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"  AI model loaded  -> {model_path}")
        except Exception as e:
            print(f"  Model load failed: {e}")
            print("  Falling back to rule-based classification.")

    # Load the fitted scaler
    scaler_path = os.path.join(BASE_DIR, "models", "hygeia_scaler.pkl")
    if JOBLIB_AVAILABLE and os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"  Scaler loaded    -> {scaler_path}")
            return
        except Exception as e:
            print(f"  Scaler load failed: {e}")

    # Fallback: fit a scaler directly from the training CSV
    if SKLEARN_AVAILABLE:
        csv_path = os.path.join(BASE_DIR, "data", "patient_vitals.csv")
        if os.path.exists(csv_path):
            print("  Fitting fallback scaler from patient_vitals.csv...")
            df = pd.read_csv(csv_path).dropna()
            df["Temp_SpO2_ratio"]  = df["Temp"]  / (df["SpO2"]  + 1e-6)
            df["Pulse_SpO2_ratio"] = df["Pulse"] / (df["SpO2"]  + 1e-6)
            df["Health_Index"] = (
                df["SpO2"] * 0.5
                - abs(df["Temp"]  - 37.0) * 10
                - abs(df["Pulse"] - 75.0) * 0.1
            )
            feature_cols = [
                "Temp", "SpO2", "Pulse",
                "Temp_SpO2_ratio", "Pulse_SpO2_ratio", "Health_Index"
            ]
            scaler = RobustScaler()
            scaler.fit(df[feature_cols].values)
            print("  Fallback scaler ready.")


# ==============================================================================
#  CLASSIFICATION
# ==============================================================================

def rule_based_classify(temp: float, spo2: float, pulse: float):
    """
    Deterministic rule-based fallback classifier.

    Used when the AI model is unavailable (e.g., TensorFlow not installed
    or model file not found). Mirrors the labelling logic used during training
    so results remain clinically consistent.

    Returns:
        (status_index: int, confidence: float)
    """
    if spo2 < 90 or temp > 40.0 or temp < 35.0 or pulse > 150 or pulse < 40:
        return 2, 0.95   # Critical
    if spo2 < 95 or temp > 38.5 or temp < 36.0 or pulse > 120 or pulse < 55:
        return 1, 0.85   # Warning
    return 0, 0.92       # Normal


def classify(temp: float, spo2: float, pulse: float):
    """
    Classify a vital sign reading using the AI model (preferred) or the
    rule-based fallback.

    The AI model receives a time-series sequence of the last 5 scaled readings
    (shape: 1 x 5 x 6). The scaler and sequence buffer are maintained
    across calls using the module-level `recent_features` deque.

    Returns:
        (status_index: int, confidence: float, status_name: str)
    """
    global recent_features

    if model is not None and scaler is not None:
        try:
            # Build the 6-feature vector for this reading
            features = np.array([[
                temp, spo2, pulse,
                temp  / (spo2  + 1e-6),
                pulse / (spo2  + 1e-6),
                spo2 * 0.5 - abs(temp - 37.0) * 10 - abs(pulse - 75.0) * 0.1
            ]])
            scaled = scaler.transform(features)[0]

            # Append to the sliding window buffer
            with buffer_lock:
                if len(recent_features) == 0:
                    # Cold start: pre-fill all 5 slots with this reading
                    for _ in range(5):
                        recent_features.append(scaled)
                else:
                    recent_features.append(scaled)
                sequence = np.array(recent_features).reshape(1, 5, 6)

            probs = model.predict(sequence, verbose=0)[0]
            idx   = int(np.argmax(probs))
            return idx, float(np.max(probs)), STATUS_NAMES[idx]

        except Exception as e:
            print(f"  Inference error: {e} – using rule-based fallback.")

    idx, conf = rule_based_classify(temp, spo2, pulse)
    return idx, conf, STATUS_NAMES[idx]


def make_reading(temp: float, spo2: float, pulse: float, source: str = "demo") -> dict:
    """
    Classify a vital sign triplet and return a complete, serialisable reading dict.
    """
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


# ==============================================================================
#  DEMO DATA GENERATOR
# ==============================================================================

# Target vital values for each simulation scenario
SIMULATION_TARGETS = {
    "normal":         (36.8, 98.0, 75.0),
    "critical_spo2":  (37.0, 85.0, 110.0),   # Hypoxaemia / low oxygen
    "critical_temp":  (41.5, 96.0, 125.0),   # High fever
    "critical_pulse": (36.9, 97.0, 165.0),   # Tachycardia
}


def _sample_loop():
    """
    Background thread that generates realistic demo vital sign readings.

    Uses a Gaussian random walk that steers 20% toward the current simulation
    target at each 5-second tick, producing smooth, realistic trends rather
    than abrupt jumps. Stops automatically when a live external device connects.
    """
    global sample_running, simulation_mode

    # Start near healthy baseline values
    temp  = random.uniform(36.5, 37.2)
    spo2  = random.uniform(96.0, 99.0)
    pulse = random.uniform(68.0, 88.0)

    # Prepare the demo session log file
    log_path = os.path.join(BASE_DIR, "data", "demo_session_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("Timestamp,Temp,SpO2,Pulse,SimulationMode\n")

    while sample_running:
        # Determine the target values for the current simulation mode
        target_temp, target_spo2, target_pulse = SIMULATION_TARGETS.get(
            simulation_mode, SIMULATION_TARGETS["normal"]
        )

        # Steer 20% toward the target and add small Gaussian noise for realism
        temp  = temp  + (target_temp  - temp)  * 0.2 + random.gauss(0, 0.05)
        spo2  = spo2  + (target_spo2  - spo2)  * 0.2 + random.gauss(0, 0.3)
        pulse = pulse + (target_pulse - pulse) * 0.2 + random.gauss(0, 1.5)

        # Hard clamp to physiologically valid bounds
        temp  = max(30.0, min(45.0, temp))
        spo2  = max(50.0, min(100.0, spo2))
        pulse = max(30.0, min(220.0, pulse))

        t, s, p = round(temp, 1), round(spo2, 0), round(pulse, 0)
        reading = make_reading(t, s, p, source="demo")

        # Append to the session log
        try:
            with open(log_path, "a") as f:
                f.write(f"{reading['timestamp']},{t},{s},{p},{simulation_mode}\n")
        except Exception:
            pass

        with buffer_lock:
            readings_buffer.append(reading)

        time.sleep(5)


def pre_populate_history():
    """
    Fill the readings buffer with 30 minutes of simulated historical data.

    This ensures the dashboard chart is immediately useful when it first loads,
    rather than starting from an empty state.
    """
    temp  = 36.8
    spo2  = 98.0
    pulse = 78.0
    now   = datetime.now()
    history = []

    for i in range(360):
        # Assign timestamps going back 30 minutes
        offset = timedelta(seconds=(360 - i) * 5)
        ts = (now - offset).isoformat()

        # Small Gaussian walk to simulate natural variation
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

    print(f"  Pre-populated {len(history)} historical readings (30 minutes).")


# ==============================================================================
#  API ROUTES
# ==============================================================================

@app.route("/")
def index():
    """Serve the web dashboard."""
    return render_template("index.html")


@app.route("/api/reading", methods=["POST"])
def receive_reading():
    """
    Accept a live vital sign reading from an external device.

    Expected JSON body:
        { "temp": 36.8, "spo2": 98, "pulse": 82 }

    When a live reading is received, the demo generator is automatically stopped.

    Returns:
        { "status": "ok", "classification": "Normal", "confidence": 0.9234 }
    """
    global live_connected, last_live_time, sample_running

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided."}), 400

    try:
        temp  = float(data["temp"])
        spo2  = float(data["spo2"])
        pulse = float(data["pulse"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Missing or invalid field: {e}"}), 400

    # Mark that a live device is connected and stop the demo generator
    live_connected = True
    last_live_time = datetime.now()
    if sample_running:
        sample_running = False

    reading = make_reading(temp, spo2, pulse, source="live")
    with buffer_lock:
        readings_buffer.append(reading)

    return jsonify({
        "status":         "ok",
        "classification": reading["status"],
        "confidence":     reading["confidence"],
    })


@app.route("/api/data")
def get_data():
    """Return all buffered readings (up to the last 360 = 30 minutes)."""
    with buffer_lock:
        return jsonify(list(readings_buffer))


@app.route("/api/current")
def get_current():
    """Return only the most recent reading."""
    with buffer_lock:
        if readings_buffer:
            return jsonify(readings_buffer[-1])
    return jsonify({})


@app.route("/api/status")
def get_status():
    """
    Return system status information.

    Includes: live device connection state, demo generator state, simulation
    mode, reading count, model load status, and classifier type.
    """
    live_alive = (
        live_connected
        and last_live_time is not None
        and (datetime.now() - last_live_time).total_seconds() < 15
    )
    return jsonify({
        "live_connected":  live_alive,
        "sample_running":  sample_running,
        "simulation_mode": simulation_mode,
        "readings_count":  len(readings_buffer),
        "model_loaded":    model is not None,
        "classifier":      "AI (TensorFlow)" if model is not None else "Rule-based",
    })


@app.route("/api/sample/start", methods=["POST"])
def start_sample():
    """Start the internal demo data generator."""
    global sample_thread, sample_running
    if not sample_running:
        sample_running = True
        sample_thread  = threading.Thread(target=_sample_loop, daemon=True)
        sample_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/sample/stop", methods=["POST"])
def stop_sample():
    """Stop the internal demo data generator."""
    global sample_running
    sample_running = False
    return jsonify({"status": "stopped"})


@app.route("/api/simulation/mode", methods=["POST"])
def set_sim_mode():
    """
    Change the active simulation scenario.

    Expected JSON body:
        { "mode": "normal" | "critical_spo2" | "critical_temp" | "critical_pulse" }
    """
    global simulation_mode
    data = request.get_json(silent=True)
    if data and "mode" in data:
        simulation_mode = data["mode"]
    return jsonify({"status": "ok", "mode": simulation_mode})


@app.route("/api/clear", methods=["POST"])
def clear_data():
    """Clear the readings buffer and reset the sequence window."""
    global recent_features
    with buffer_lock:
        readings_buffer.clear()
        recent_features.clear()
    return jsonify({"status": "cleared"})


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HYGEIA AI Server")
    parser.add_argument("--hardware", action="store_true", help="Start without demo data, wait for hardware")
    args = parser.parse_args()

    print()
    print("=" * 55)
    print("  HYGEIA AI – Patient Health Monitoring System")
    print("=" * 55)
    print()

    print("Loading AI model and scaler...")
    load_model_and_scaler()
    print()

    print("Pre-populating 30-minute history buffer...")
    pre_populate_history()
    print()

    if not args.hardware:
        # Auto-start the demo generator
        sample_running = True
        sample_thread  = threading.Thread(target=_sample_loop, daemon=True)
        sample_thread.start()
        print("Demo mode is active. The generator will stop when a live device connects.")
    else:
        print("Hardware mode is active. Waiting for live device connection...")
        
    print()
    print("=" * 55)
    print("  Dashboard : http://localhost:5000")
    print("  API POST  : http://<your-pc-ip>:5000/api/reading")
    print("=" * 55)
    print()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
