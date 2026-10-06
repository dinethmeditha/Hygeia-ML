# HYGEIA AI – Patient Health Monitoring System
### Essentials of Artificial Intelligence | Group Assignment | Stage 3 – Final Presentation
### IT3182 / CS22032 / CS22023 | Week 10

---

> **Assignment Requirement:** Apply at least three AI-related concepts, demonstrate AI-based reasoning/prediction/classification, and each member must present a clearly defined section.

---

## Presentation Structure & Member Responsibilities

| Part | Member | Topic Coverage |
|------|--------|----------------|
| **Part 1** | Member 1 | Introduction, Problem Background, Objectives |
| **Part 2** | Member 2 | AI Techniques & System Design |
| **Part 3** | Member 3 | Data Collection, ML Pipeline & Implementation |
| **Part 4** | Member 4 | Testing, Results, Ethics & Conclusion |

---

---

# PART 1 – Introduction & Problem Background
### Presenter: Member 1

---

## 1.1 Project Title

> **HYGEIA AI – A Real-Time AI-Powered Patient Health Monitoring System**

---

## 1.2 Real-World Problem

In clinical and home care environments, medical staff cannot **continuously watch every patient's vital signs** at all times. Critical deterioration in patient health — a sudden drop in oxygen saturation, a high fever, or an abnormal pulse — can go unnoticed for several minutes or longer. Every minute of delay in response time can be life-threatening.

Current monitoring systems:
- Alert only when a **single threshold is breached** (reactive, not predictive).
- Do not consider **trends** — e.g., a slowly dropping SpO2 that has not crossed the limit yet.
- Require constant **manual observation** by nurses or caregivers.
- Produce high numbers of **false alarms** due to simplistic rules.

---

## 1.3 Who Is Affected?

| Stakeholder | Impact |
|---|---|
| **ICU / Ward Patients** | Delayed detection of deterioration |
| **Nurses & Doctors** | Overwhelmed by manual monitoring and false alarms |
| **Elderly / Home-Care Patients** | No reliable continuous remote monitoring solution |
| **Hospital Administration** | Increased liability and resource pressure |

---

## 1.4 Why AI Is the Right Solution

Traditional monitoring uses **fixed threshold rules** (e.g., "alert if SpO2 < 90%"). This approach:
- Misses **gradual deterioration** happening within normal-looking ranges.
- Cannot learn from **historical patterns** of a patient.

AI (specifically deep learning time-series classification) can:
- **Analyse trends** across the last 25 seconds of readings.
- **Classify risk proactively** before a hard threshold is crossed.
- Continuously **improve** with more patient data.

---

## 1.5 Objectives of the HYGEIA AI System

1. **Monitor** patient vital signs (Temperature, SpO2, Pulse Rate) in real time.
2. **Classify** patient risk level as **Normal**, **Warning**, or **Critical** using an AI model.
3. **Predict trend-based deterioration** using sliding window time-series analysis.
4. **Display** a high-contrast, clinically readable live dashboard for caretakers.
5. **Log** all patient session data for future analysis and model improvement.
6. **Simulate** different critical patient scenarios for testing and demonstration.

---

## 1.6 Target Users

- **ICU nurses and ward staff** monitoring multiple patients simultaneously.
- **Home-care workers** looking after elderly or post-surgery patients.
- **Hospital administrators** who need audit-ready session logs.

---

---

# PART 2 – AI Techniques & System Design
### Presenter: Member 2

---

## 2.1 AI Techniques Applied (3+ as Required)

### Technique 1 – Supervised Classification with Neural Networks
**What:** A deep learning model trained on labelled patient vital sign data.  
**Why:** The problem is a multi-class classification task (Normal / Warning / Critical). Neural networks excel at learning non-linear boundaries between classes.  
**How it works in HYGEIA:** The model receives 5 consecutive readings and outputs a probability distribution over 3 health status classes.

---

### Technique 2 – Time-Series Learning (Sliding Window)
**What:** Sequential data is fed to the model as a window of the last 5 readings (25 seconds of history).  
**Why:** Physiological deterioration is a temporal process. A single reading at 94% SpO2 is "Warning", but 5 consecutive readings all trending downward from 98% to 94% is a much more dangerous pattern. The AI can detect this; fixed rules cannot.  
**How it works in HYGEIA:**

```
Reading 1 -> Reading 2 -> Reading 3 -> Reading 4 -> Reading 5
     |            |            |            |            |
[ Feature Vector x 6 ] x 5 timesteps = Input shape (5, 6) -> Model -> Risk Label
```

---

### Technique 3 – Rule-Based Expert System (Fallback Engine)
**What:** A deterministic decision tree of clinical threshold rules.  
**Why:** AI models can fail to load or may encounter out-of-distribution inputs. A safety-critical system must always be able to classify, even without the neural network. This is the "expert knowledge" fallback.  
**How it works in HYGEIA:**

```
IF SpO2 < 90% OR Temp > 40C OR Temp < 35C OR Pulse > 150 OR Pulse < 40
  -> CRITICAL  (confidence: 0.95)
ELSE IF SpO2 < 95% OR Temp > 38.5C OR Pulse > 120
  -> WARNING   (confidence: 0.85)
ELSE
  -> NORMAL    (confidence: 0.92)
```

---

### Technique 4 – Feature Engineering (Derived AI Inputs)
**What:** Transforming 3 raw sensor inputs into 6 engineered features that amplify anomaly signals.  
**Why:** The raw values alone may not expose the interaction between vitals (e.g., high pulse combined with low SpO2 is more dangerous than either alone). Engineered features encode these relationships.

| Engineered Feature | Formula | Clinical Meaning |
|---|---|---|
| `Temp_SpO2_ratio` | `Temp / SpO2` | Amplifies fever + hypoxia co-occurrence |
| `Pulse_SpO2_ratio` | `Pulse / SpO2` | Amplifies tachycardia + low oxygen |
| `Health_Index` | `SpO2 x 0.5 - abs(Temp-37) x 10 - abs(Pulse-75) x 0.1` | Composite clinical wellness score |

---

## 2.2 System Architecture

```
+------------------------------------------------------------------+
|  DATA INPUT LAYER                                                |
|  +------------------------------------------------------------+  |
|  |  data_streamer.py (patient_vitals.csv row-by-row)         |  |
|  |  OR Internal Demo Generator (normal / critical modes)     |  |
|  |  OR Live External Device -> POST /api/reading             |  |
|  +---------------------------+--------------------------------+  |
+------------------------------+-----------------------------------+
                               | JSON: { temp, spo2, pulse }
                               v
+------------------------------------------------------------------+
|  AI PROCESSING LAYER (server.py)                                 |
|                                                                  |
|  [Feature Engineering] -> [RobustScaler] -> [Sequence Buffer]   |
|                                                  |               |
|                          +-----------------------v-----------+   |
|                          |  Conv1D + BiLSTM Neural Network  |   |
|                          |  Input:  (1, 5, 6)               |   |
|                          |  Output: P(Normal/Warning/Crit.) |   |
|                          +-----------------------+-----------+   |
|                                                  |               |
|  [30-Min Rolling Buffer <- append reading]       |               |
+--------------------------------------------------+---------------+
                                                   |
                                                   v
+------------------------------------------------------------------+
|  PRESENTATION LAYER  (http://localhost:5000)                     |
|  Live Vital Cards | AI Risk Badge | 30-Min Chart | Sim Controls  |
+------------------------------------------------------------------+
```

---

## 2.3 System Input / Output Design

| | Detail |
|---|---|
| **Inputs** | Temperature (C), SpO2 (%), Pulse Rate (BPM) |
| **Processing** | Feature engineering -> scaling -> 5-step sequence -> neural network |
| **Outputs** | Risk status (Normal / Warning / Critical), Confidence %, timestamp |
| **Storage** | Rolling 360-reading buffer + `data/demo_session_log.csv` |

---

## 2.4 Tools & Technologies

| Component | Technology |
|---|---|
| AI Model | TensorFlow 2.13 / Keras |
| Data Scaling | scikit-learn RobustScaler |
| Backend Server | Python 3.8 + Flask |
| Frontend | HTML5 / CSS3 / JavaScript + Chart.js |
| Data Handling | pandas, numpy |
| Model Persistence | h5py (Keras `.h5`), joblib (`.pkl`) |

---

---

# PART 3 – Data Collection, ML Pipeline & Implementation
### Presenter: Member 3

---

## 3.1 Dataset

**File:** `data/patient_vitals.csv`  
**Content:** Collected patient vital sign readings, each row representing one monitoring snapshot.

| Column | Type | Example |
|--------|------|---------|
| `Temp` | float | 36.8 |
| `SpO2` | int | 98 |
| `Pulse` | int | 74 |
| `Status` | string | Normal |

---

## 3.2 Data Validation Pipeline

```
Raw CSV
   |
   v
Step 1: Coerce all columns to numeric  (non-numeric -> NaN)
   |
   v
Step 2: Hard Physiological Bounds Filter
   |-- Temp:   30C  < T  < 45C
   |-- SpO2:    0%  < S  <= 100%
   +-- Pulse:   0   < P  < 200
   |
   v
Step 3: Drop remaining NaN rows
   |
   v
Step 4: Reset index
   |
   v
   Clean DataFrame -> Feature Engineering -> Model Training
```

**Why RobustScaler?**
Standard scalers (MinMax, Standard) are sensitive to outliers. Patient data contains natural spikes. `RobustScaler` uses median and IQR, making it robust to extreme but valid readings.

---

## 3.3 Sequence (Sliding Window) Creation

```python
SEQ_LEN = 5  # 5 readings x 5 seconds = 25 seconds of history

for i in range(len(X) - SEQ_LEN + 1):
    X_seq.append(X[i : i + SEQ_LEN])   # shape: (5, 6)
    y_seq.append(y[i + SEQ_LEN - 1])   # label of the final step
```

**Why 5 steps?**
5 readings x 5-second intervals = **25 seconds of trend context**. Short enough to respond quickly to sudden changes; long enough to smooth out single noisy readings.

---

## 3.4 Neural Network Architecture

```
Input:  shape (5, 6)
   |
   v
Conv1D(64 filters, kernel=2, ReLU)     <- Local pattern detection
BatchNormalization
Dropout(0.2)
   |
   v
Conv1D(128 filters, kernel=2, ReLU)    <- Deeper pattern extraction
BatchNormalization
Dropout(0.2)
   |
   v
Bidirectional LSTM(64 units)            <- Forward + backward temporal context
Dropout(0.3)
   |
   v
Bidirectional LSTM(32 units)            <- Fine-grained temporal refinement
Dropout(0.3)
   |
   v
Dense(64, ReLU)
Dropout(0.3)
   |
   v
Dense(3, Softmax)                       <- 3 output classes
   |
   v
Output: [ P(Normal),  P(Warning),  P(Critical) ]
```

**Why Conv1D before LSTM?**
Conv1D layers act as local feature detectors — they learn short patterns like "SpO2 dropped 2 steps in a row". LSTM layers then capture longer-range sequential dependencies across those detected patterns.

**Why Bidirectional LSTM?**
Standard LSTM only sees the sequence forward. Bidirectional LSTM also processes it backwards, doubling the temporal context and improving classification accuracy.

---

## 3.5 Training Configuration

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Max Epochs | 150 |
| Batch Size | 16 |
| Train/Test Split | 80% / 20% |
| EarlyStopping Patience | 20 epochs |
| ReduceLROnPlateau | Factor 0.5, Patience 10 |

---

## 3.6 Real-Time Inference Flow (Server-Side)

```
New Reading (temp, spo2, pulse)
         |
         v
Feature Engineering  ->  6 features
         |
         v
RobustScaler.transform()
         |
         v
Append to deque(maxlen=5)     <- stateful sequence buffer
         |
         v
Reshape to (1, 5, 6)
         |
         v
model.predict()  ->  softmax probabilities
         |
         v
argmax -> status index -> "Normal" / "Warning" / "Critical"
         |
         v
Append to 30-min rolling buffer -> send to dashboard
```

---

## 3.7 Web Dashboard Components

The dashboard (`templates/index.html`) is a single-page app served by Flask:

| Component | Description |
|---|---|
| Vital Cards | Large-font Temp / SpO2 / Pulse with colour coding |
| AI Risk Badge | Current status and confidence progress bar |
| 30-Min Trend Chart | Scrolling Chart.js line chart (360 readings) |
| Simulation Controller | 4 buttons to switch patient scenario in real time |

JS polls `/api/data` every 3 seconds and updates the UI without page reload.

---

---

# PART 4 – Testing, Results, Ethics & Conclusion
### Presenter: Member 4

---

## 4.1 Testing Strategy

| Layer | Method | Purpose |
|---|---|---|
| **Model Testing** | Held-out 20% test set | Measure classification accuracy and confidence |
| **Functional Testing** | `start_demo.bat` end-to-end | Verify server -> streamer -> dashboard pipeline |
| **Scenario Testing** | Simulation Controller (4 modes) | Verify critical case detection in real time |
| **Fallback Testing** | Rename model file, re-run | Verify rule-based engine activates correctly |

---

## 4.2 Model Performance Results

| Metric | Value |
|---|---|
| Overall Accuracy | **100%** on test split |
| Mean Confidence | **~99.97%** |
| Architecture | Conv1D x 2 + BiLSTM x 2 |
| Sequence Length | 5 steps (25 seconds) |

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 1.00 | 1.00 | 1.00 | 38 |
| Warning | 1.00 | 1.00 | 1.00 | 1 |
| Critical | 0.00 | 0.00 | 0.00 | 0 (no test samples) |

> *Note: Critical class validation was done through simulation testing only, since the small dataset's 20% test split contained no Critical samples. The rule-based fallback handles this case with 95% confidence.*

---

## 4.3 Simulation Scenario Test Results

| Simulation Mode | Target Values | Dashboard Behaviour | Result |
|---|---|---|---|
| Normal Patient | SpO2 ~98%, Temp ~36.8C, Pulse ~75 | Status: NORMAL (green) | PASS |
| Drop SpO2 | SpO2 -> ~85% | Transitions Warning -> Critical within ~30s | PASS |
| Spike Temp | Temp -> ~41.5C | Transitions to Critical | PASS |
| High Pulse | Pulse -> ~165 BPM | Transitions to Critical | PASS |
| Return to Normal | All reset to normal targets | Recovers to NORMAL | PASS |

---

## 4.4 Sample System Output

**Normal State:**
```
Temperature : 36.8 C    SpO2 : 98%    Pulse : 74 BPM
AI Status   : NORMAL    Confidence : 99.9%
```

**After switching to Drop SpO2 simulation (~30 seconds):**
```
Temperature : 36.9 C    SpO2 : 85%    Pulse : 112 BPM
AI Status   : CRITICAL    Confidence : 99.7%
```

---

## 4.5 Limitations of the Solution

1. **Small Training Dataset** — Limited number of patient readings. A larger, more diverse dataset would improve generalisation across different patient demographics.

2. **No Critical Samples in Test Set** — Because Critical events are rare, the held-out test split contained zero Critical instances. Critical-class performance is validated through simulation only.

3. **5-Second Polling Interval** — The 5-second update cycle may be too slow for rapidly evolving acute events (e.g., cardiac arrest).

4. **Single Patient Focus** — The current system monitors one patient session at a time. A real deployment needs multi-patient management.

5. **Software Simulation Only** — The current demo runs entirely in software. Integration with physical medical sensors is required for clinical deployment.

---

## 4.6 Ethical & Social Considerations

| Issue | HYGEIA AI Approach |
|---|---|
| **Data Privacy** | All data processed locally. No external cloud transmission. In production, session logs must be encrypted. |
| **Algorithmic Bias** | Model trained on limited dataset. Clinical validation across diverse demographics (age, ethnicity, conditions) is mandatory before real use. |
| **Accuracy & Reliability** | Rule-based fallback guarantees a classification is always produced. AI output is decision support, not a medical diagnosis. |
| **Human Responsibility** | All alerts must be reviewed by qualified medical personnel. The AI is an assistant, not a replacement for clinical judgement. |
| **Over-Reliance Risk** | Staff must not become complacent. The system complements, not replaces, routine manual assessments. |
| **Misuse** | Dashboard access must be restricted. Authentication and access control are required before any clinical deployment. |

---

## 4.7 Future Works

- **Automated Critical Alert Notifications** — SMS/email/push alerts to doctors when a patient reaches Critical status, regardless of whether anyone is watching the dashboard.
- **Cloud Deployment** — Host on AWS or Google Cloud for hospital-wide access.
- **Multi-Patient Management** — Individual profiles, history, and dashboards per patient.
- **PDF/CSV Report Export** — Clinical-grade vital sign reports for medical records and audit.
- **Anomaly Detection Pre-Layer** — Unsupervised Autoencoder to detect rare critical patterns outside the training distribution.

---

## 4.8 Conclusion

**HYGEIA AI** successfully demonstrates how Artificial Intelligence concepts can be applied to a critical real-world healthcare problem. The project integrates:

1. **Supervised Deep Learning** (Conv1D + BiLSTM neural network) — patient risk classification.
2. **Time-Series Analysis** (sliding window sequences) — trend-based deterioration detection.
3. **Rule-Based Expert System** — robust fallback classification engine.
4. **Feature Engineering** — extracting richer clinically meaningful signals from raw data.

The system achieves real-time classification, provides a clinical-grade dashboard, logs all session data, and supports interactive scenario simulation — fulfilling all assignment requirements.

---

## Individual Contribution Table

| Member | Presentation Section | Report Sections Covered |
|---|---|---|
| **Member 1** | Introduction & Problem Background | Title page, Abstract, Introduction, Problem Background, Problem Statement, Objectives, Target Users |
| **Member 2** | AI Techniques & System Design | Selected AI Techniques, Justification of Techniques, System Architecture Diagram, Tools & Technologies |
| **Member 3** | Data Collection, ML & Implementation | Dataset Details, Preprocessing Pipeline, Model Architecture, Implementation Details, Dashboard |
| **Member 4** | Testing, Ethics & Conclusion | Testing & Results, Sample Outputs, Limitations, Ethical Considerations, Future Works, Conclusion |

---

## References

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
4. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
5. World Health Organization. (2020). *Pulse Oximetry Training Manual*. WHO Press.
6. Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM and other neural network architectures. *Neural Networks*, 18(5-6), 602–610.

---

*HYGEIA AI | KDU Semester 4 | Essentials of Artificial Intelligence – Group Assignment | Stage 3 Final Submission*
