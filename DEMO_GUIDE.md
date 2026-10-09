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
3. Begin streaming patient data or run the interactive demo

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
| [README.md](README.md) | Section 1: Full project overview, objectives |
| [README.md](README.md) | Section 3: Architecture diagram |
| [data/patient_vitals.csv](data/patient_vitals.csv) | The collected patient data used for training |

### Talking Points & File Walk-through

**1. Open README.md → Section 1 (Project Overview)**
- Explain the real-world problem: nurses cannot watch every patient 24/7.
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
| [server.py](server.py) | `classify()` function, `rule_based_classify()` |
| [hygeia_local.py](hygeia_local.py) | Feature engineering block |

### Talking Points & File Walk-through

**1. Discuss AI Techniques**
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
| [results/classification_report.txt](results/classification_report.txt) | Model accuracy and per-class results |
| [results/final_metrics_summary.txt](results/final_metrics_summary.txt) | Overall accuracy, confidence scores |
| [results/](results/) | Training loss/accuracy plots and confusion matrix |

### Talking Points & File Walk-through

**1. Open hygeia_local.py → Data Validation block (top of file)**
- Show the range filter: `30 < Temp < 45`, `0 < SpO2 <= 100`, `0 < Pulse < 200`.
- Explain: "Raw data can have sensor errors. We validate every reading before it ever touches the model."

**2. Open hygeia_local.py → Sequence Creation block**
- Show the sliding window loop.
- Explain: "Instead of classifying a single snapshot, we feed 5 consecutive readings at once — 25 seconds of history. This is how the model detects trends."

**3. Open hygeia_local.py → Model Build section**
- Walk through each layer: Conv1D → Conv1D → BiLSTM → BiLSTM → Dense → Softmax.
- Explain why Conv1D comes before LSTM (local pattern extraction before temporal learning).

**4. Open results/classification_report.txt**
- Show the overall accuracy and mean confidence.

**5. Open results/ folder**
- Open the training loss curve plot (`advanced_training_metrics.png`) — show the model converging cleanly.
- Open the confusion matrix (`comprehensive_evaluation_heatmaps.png`) — show the classification results.

### Key Message for Member 3
> "The ML pipeline goes from raw CSV data through strict validation, feature engineering, sequence creation, and a Conv1D + BiLSTM deep learning model — achieving high accuracy and confidence."

---

---

## MEMBER 4 – Testing, Results, Live Demo & Ethics

### Files to Open & Show

| File | What to Show |
|------|-------------|
| [data/demo_session_log.csv](data/demo_session_log.csv) | Live-logged session data from demo |
| [testing/](testing/) | Folder showing the test scripts we built |
| [http://localhost:5000](http://localhost:5000) | Live dashboard – run the demo here |

### Talking Points & Live Demo Steps

**1. Show our Testing approach**
- Open the `testing/` folder and explain we built automated API tests (`test_api_endpoints.py`), a dummy data generator (`generate_dummy_data.py`), and unit tests for the classifier (`test_classification.py`).

**2. LIVE DEMO – Dashboard (http://localhost:5000)**

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

**3. Discuss Ethical Considerations**
- Walk through: data privacy (local only processing), bias (demographic limitation of training data), human responsibility (AI is decision support, not diagnosis), and over-reliance risk.

**4. Wrap up with Future Works (README Section 15)**
- Notification alerts for Critical status.
- Multi-patient management.
- Cloud deployment.

### Key Message for Member 4
> "The system passed all test scenarios and has automated test coverage. The live demo shows the AI correctly detecting critical patient conditions in real-time, with all data automatically logged. Ethical safeguards are built into the design — this is a tool for clinicians, not a replacement."

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

### AI Model & Testing
| File | Role |
|------|------|
| [models/](models/) | Contains trained `.h5` model and `.pkl` scaler |
| [testing/](testing/) | Automated test scripts and data generators |

### Data & Results
| File | Role |
|------|------|
| [data/patient_vitals.csv](data/patient_vitals.csv) | Training dataset |
| [data/demo_session_log.csv](data/demo_session_log.csv) | Auto-logged demo session |
| [results/](results/) | Training plots, confusion matrix, accuracy reports |

### Documentation
| File | Role |
|------|------|
| [README.md](README.md) | Full project documentation |
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | This file – demo & presenter guide |
| [docs/Essentials_of_AI_Group_Assignment.pdf](docs/Essentials_of_AI_Group_Assignment.pdf) | Assignment criteria |

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
