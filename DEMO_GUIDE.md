# HYGEIA AI – Live Presentation & Demo Guide
### Essentials of AI | Stage 3 – Final Presentation | Demo Day

> **This file is the presenter's cheat-sheet.**
> Each member has a clear list of files to open, talk points, and demo steps.
> Keep this open side-by-side with the files during the presentation.

---

## Quick Launch (Before Presenting – Run This First)

Double-click **[start_demo.bat](start_demo.bat)** to automatically:
1. Start the AI Flask server
2. Open the dashboard in the browser at http://localhost:5000
3. Begin streaming patient data

> If the bat file does not work, run these manually in separate terminals:
> ```
> python server.py
> python data_streamer.py
> ```

---

---

## MEMBER 1 – Project Overview & Problem Background

### Files to Open & Show

| File | What to Show |
|------|-------------|
| [README.md](README.md) | Full project overview, objectives, architecture diagram |
| [data/patient_vitals.csv](data/patient_vitals.csv) | The collected patient data used for training |

### Talking Points & File Walk-through

**1. Open README.md → Section 1 (Project Overview)**
- Explain the real-world problem: nurses cannot watch every patient 24/7.
- Show the "who is affected" table.
- Emphasise that traditional monitors only fire when a threshold is crossed — HYGEIA AI detects *trends*.

**2. Open README.md → Section 3 (System Architecture diagram)**
- Walk through the ASCII architecture: Data Source → Server → AI Model → Dashboard.
- Point to each layer and explain what happens there.

**3. Open data/patient_vitals.csv**
- Show the 3 raw columns: `Temp`, `SpO2`, `Pulse`.
- Explain that this is real collected data, and it is the training input for the AI.

**4. Open README.md → Section 7 (Risk Classification Rules)**
- Show the threshold table (Normal / Warning / Critical criteria).
- Explain this is also used as the expert system fallback.

### Key Message for Member 1
> "We identified a real clinical problem, collected patient data, and designed an AI system to detect health deterioration before it becomes a crisis."

---

---

## MEMBER 2 – AI Techniques & System Design

### Files to Open & Show

| File | What to Show |
|------|-------------|
| [README.md](README.md) | Section 5 – ML Methodology, Section 6 – Classification Rules |
| [server.py](server.py) | `classify()` function (line ~128), `rule_based_classify()` (line ~119) |
| [hygeia_local.py](hygeia_local.py) | Feature engineering block (line ~60–80) |
| [PRESENTATION.md](PRESENTATION.md) | Part 2 – AI Techniques section (architecture diagram) |

### Talking Points & File Walk-through

**1. Open PRESENTATION.md → Part 2 (AI Techniques)**
- Walk through all 4 AI techniques: Neural Network, Time-Series Sliding Window, Rule-Based Expert System, Feature Engineering.
- For each one: What is it? Why did we use it? How does it work in HYGEIA?

**2. Open hygeia_local.py → Feature Engineering section**
- Show lines where `Temp_SpO2_ratio`, `Pulse_SpO2_ratio`, and `Health_Index` are calculated.
- Explain why 3 raw inputs become 6 engineered features.

**3. Open server.py → `rule_based_classify()` function**
- Show the if/elif/else decision tree.
- Explain: "This is our expert knowledge — if the AI model ever fails to load, this guarantees the system always classifies correctly."

**4. Open server.py → `classify()` function**
- Show how it checks if the model is available first, falls back to rule-based if not.
- Explain the stateful `deque(maxlen=5)` — this is how we maintain the time-series window.

**5. Show the Architecture Diagram (PRESENTATION.md or README.md)**
- Walk through the system design: Input → Feature Engineering → Scaler → Sequence → Model → Output → Dashboard.

### Key Message for Member 2
> "We applied four AI concepts: supervised neural network classification, time-series sliding window learning, a rule-based expert system fallback, and feature engineering — all working together to make the system accurate and reliable."

---

---

## MEMBER 3 – Data Pipeline, ML & Implementation

### Files to Open & Show

| File | What to Show |
|------|-------------|
| [hygeia_local.py](hygeia_local.py) | Full training script – validation, sequences, model build, training |
| [README.md](README.md) | Section 5 – ML Methodology |
| [classification_report.txt](classification_report.txt) | Model accuracy and per-class results |
| [final_metrics_summary.txt](final_metrics_summary.txt) | 100% accuracy, confidence scores |
| [HYGEIA_AI_FINAL_RESULTS/](HYGEIA_AI_FINAL_RESULTS/) | Training loss/accuracy plots and confusion matrix |

### Talking Points & File Walk-through

**1. Open hygeia_local.py → Data Validation block (top of file)**
- Show the range filter: `30 < Temp < 45`, `0 < SpO2 <= 100`, `0 < Pulse < 200`.
- Explain: "Raw data can have sensor errors. We validate every reading before it ever touches the model."

**2. Open hygeia_local.py → Feature Engineering block**
- Show where the 3 raw features become 6.
- Show where `RobustScaler` is fitted and saved as `hygeia_scaler.pkl`.

**3. Open hygeia_local.py → Sequence Creation block**
- Show the sliding window loop: `X[i : i + SEQ_LEN]`.
- Explain: "Instead of classifying a single snapshot, we feed 5 consecutive readings at once — 25 seconds of history. This is how the model detects trends."

**4. Open hygeia_local.py → Model Build section**
- Walk through each layer: Conv1D → Conv1D → BiLSTM → BiLSTM → Dense → Softmax.
- Explain why Conv1D comes before LSTM (local pattern extraction before temporal learning).

**5. Open classification_report.txt**
- Show: Normal 100%, Warning 100%, Overall Accuracy 100%, Mean Confidence 1.0000.
- Mention that Critical class had no test samples (data imbalance — rare events).

**6. Open HYGEIA_AI_FINAL_RESULTS/ folder**
- Open the training loss curve plot — show the model converging cleanly.
- Open the confusion matrix — show that no Normal samples were misclassified.

### Key Message for Member 3
> "The ML pipeline goes from raw CSV data through strict validation, feature engineering, sequence creation, and a Conv1D + BiLSTM deep learning model — achieving 100% accuracy with 99.97% mean confidence."

---

---

## MEMBER 4 – Testing, Results, Live Demo & Ethics

### Files to Open & Show

| File | What to Show |
|------|-------------|
| [data/demo_session_log.csv](data/demo_session_log.csv) | Live-logged session data from demo |
| [data/patient_vitals.csv](data/patient_vitals.csv) | Original training data |
| [final_metrics_summary.txt](final_metrics_summary.txt) | Test results summary |
| [PRESENTATION.md](PRESENTATION.md) | Part 4 – Testing tables, ethics table |
| [http://localhost:5000](http://localhost:5000) | Live dashboard – run the demo here |

### Talking Points & Live Demo Steps

**1. Open PRESENTATION.md → Part 4 (Testing Strategy table)**
- Explain the 3 testing layers: Model testing, functional end-to-end, scenario testing.

**2. Open final_metrics_summary.txt**
- Read out the key numbers: 100% accuracy, 99.97% confidence.

**3. LIVE DEMO – Dashboard (http://localhost:5000)**

> **Demo Script (step by step):**

```
STEP 1 – Normal Mode (already running)
   - Point at the three vital cards: Temp ~36.8C, SpO2 ~98%, Pulse ~75 BPM
   - Show the AI Status: NORMAL (green)
   - Show the 30-minute trend chart scrolling in real time

STEP 2 – Trigger Drop SpO2 (critical scenario)
   - Click the "Drop SpO2" button in the Simulation Controller
   - Wait ~20-30 seconds
   - Watch SpO2 fall from 98% -> 90% -> 85%
   - Watch the AI Status transition: NORMAL -> WARNING -> CRITICAL (red)
   - Explain: "The AI detected the downward trend before SpO2 crossed 90%"

STEP 3 – Trigger Spike Temp
   - Click "Spike Temp" button
   - Wait ~20 seconds, watch Temp rise toward 40-41C
   - Show status turn to CRITICAL

STEP 4 – Return to Normal
   - Click "Normal Patient" button
   - Watch the vitals and status gradually recover
   - Emphasise: "The model tracks recovery just as well as deterioration"

STEP 5 – Show the CSV log
   - Open data/demo_session_log.csv
   - Show real rows that were just recorded by the demo
   - Explain: "Every reading is automatically logged with a timestamp and mode"
```

**4. Open PRESENTATION.md → Part 4 (Ethical Considerations table)**
- Walk through: data privacy (local only), bias (demographic limitation), human responsibility (AI is decision support, not diagnosis), over-reliance risk.

**5. Wrap up with Future Works**
- Notification alerts for Critical status.
- Multi-patient management.
- Cloud deployment.

### Key Message for Member 4
> "The system passed all test scenarios. The live demo shows the AI correctly detecting critical patient conditions in real-time, with all data automatically logged. Ethical safeguards are built into the design — this is a tool for clinicians, not a replacement."

---

---

## All Project Files at a Glance

### Core System
| File | Role |
|------|------|
| [server.py](server.py) | Flask AI server – main runtime |
| [templates/index.html](templates/index.html) | Web dashboard (frontend SPA) |
| [hygeia_local.py](hygeia_local.py) | Model training script |
| [data_streamer.py](data_streamer.py) | Streams CSV data to server API |
| [start_demo.bat](start_demo.bat) | One-click launcher |

### AI Model & Scaler
| File | Role |
|------|------|
| [hygeia_ai_model_final.h5](hygeia_ai_model_final.h5) | Trained Keras neural network |
| [hygeia_scaler.pkl](hygeia_scaler.pkl) | Fitted RobustScaler |

### Data
| File | Role |
|------|------|
| [data/patient_vitals.csv](data/patient_vitals.csv) | Training dataset |
| [data/demo_session_log.csv](data/demo_session_log.csv) | Auto-logged demo session |

### Results & Reports
| File | Role |
|------|------|
| [classification_report.txt](classification_report.txt) | Per-class precision/recall/F1 |
| [final_metrics_summary.txt](final_metrics_summary.txt) | Overall accuracy summary |
| [HYGEIA_AI_FINAL_RESULTS/](HYGEIA_AI_FINAL_RESULTS/) | Training plots & confusion matrix |

### Documentation
| File | Role |
|------|------|
| [README.md](README.md) | Full project documentation |
| [PRESENTATION.md](PRESENTATION.md) | Presentation content (4 parts) |
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | This file – demo & presenter guide |

---

## Emergency Fallback (If Server Won't Start)

```bash
# Kill any existing server on port 5000
netstat -ano | findstr :5000
taskkill /PID <pid_number> /F

# Restart with encoding fix
set PYTHONIOENCODING=utf-8
python server.py
```

---

*HYGEIA AI | KDU Semester 4 | Essentials of Artificial Intelligence – Stage 3 Final Presentation*
