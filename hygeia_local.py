"""
🧤 HYGEIA GLOVE – Model Training Script
========================================
Trains the Conv1D + BiLSTM health classification model on
Hygeia Glove sensor data (glove_readings.csv) and saves:
  - hygeia_ai_model_final.h5   (Keras model)
  - hygeia_scaler.pkl          (fitted RobustScaler for the server)

Run:  python hygeia_local.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, LSTM, Dense, Dropout,
                                      BatchNormalization, Bidirectional, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(BASE_DIR, "data", "glove_readings.csv")
MODEL_OUT   = os.path.join(BASE_DIR, "hygeia_ai_model_final.h5")
SCALER_OUT  = os.path.join(BASE_DIR, "hygeia_scaler.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "HYGEIA_AI_FINAL_RESULTS")
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)

print("✅ HYGEIA Model Training Script")
print(f"📊 TensorFlow  : {tf.__version__}")
print(f"🧠 GPU         : {len(tf.config.list_physical_devices('GPU')) > 0}")

# ── 1. Load Glove-Only Data ────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  📁 LOADING HYGEIA GLOVE DATA")
print("═" * 60)

df = pd.read_csv(CSV_PATH)
for col in ['Temp', 'SpO2', 'Pulse']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[
    (df['Temp']  > 30)  & (df['Temp']  < 45) &
    (df['SpO2']  > 0)   & (df['SpO2']  <= 100) &
    (df['Pulse'] > 0)   & (df['Pulse'] < 200)
].dropna().reset_index(drop=True)

print(f"✅ {len(df)} valid glove readings")
print(f"   Temp  : {df['Temp'].min():.1f}°C – {df['Temp'].max():.1f}°C")
print(f"   SpO2  : {df['SpO2'].min():.0f}% – {df['SpO2'].max():.0f}%")
print(f"   Pulse : {df['Pulse'].min():.0f} – {df['Pulse'].max():.0f} BPM")

# ── 2. Feature Engineering & Labels ───────────────────────────────────────────
def label(row):
    t, s, p = row['Temp'], row['SpO2'], row['Pulse']
    if s < 90 or t > 40.0 or t < 35.0 or p > 150 or p < 40: return 2  # Critical
    if s < 95 or t > 38.5 or t < 36.0 or p > 120 or p < 55: return 1  # Warning
    return 0  # Normal

df['Status']           = df.apply(label, axis=1)
df['Temp_SpO2_ratio']  = df['Temp']  / (df['SpO2']  + 1e-6)
df['Pulse_SpO2_ratio'] = df['Pulse'] / (df['SpO2']  + 1e-6)
df['Health_Index']     = (
    df['SpO2'] * 0.5
    - abs(df['Temp']  - 37.0) * 10
    - abs(df['Pulse'] - 75.0) * 0.1
)

STATUS_NAMES = {0: 'Normal', 1: 'Warning', 2: 'Critical'}
print("\n📊 Health Status Distribution:")
for k, v in STATUS_NAMES.items():
    n = int((df['Status'] == k).sum())
    print(f"   {v:8s}: {n:4d} ({100*n/len(df):.1f}%)")

# ── 3. Preprocessing ───────────────────────────────────────────────────────────
FEATURE_COLS = ['Temp', 'SpO2', 'Pulse', 'Temp_SpO2_ratio', 'Pulse_SpO2_ratio', 'Health_Index']
X = df[FEATURE_COLS].values
y = df['Status'].values

scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for the Flask server to use
joblib.dump(scaler, SCALER_OUT)
print(f"\n✅ Scaler saved → {SCALER_OUT}")

# Reshape to (N, 1, 6) for Conv1D/LSTM
X_seq = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 4. Model ───────────────────────────────────────────────────────────────────
def build_model(input_shape, n_classes=3):
    inputs = Input(shape=input_shape, name='vital_signs')

    x = Conv1D(64,  kernel_size=1, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, kernel_size=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64,  activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(n_classes, activation='softmax', name='status')(x)
    return Model(inputs, outputs, name='HYGEIA_AI')

model = build_model((X_train.shape[1], X_train.shape[2]))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── 5. Training ────────────────────────────────────────────────────────────────
cbs = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_OUT, monitor='val_accuracy', save_best_only=True, verbose=1),
]

print("\n🚀 Training …")
history = model.fit(
    X_train, y_train,
    epochs=150, batch_size=16,
    validation_split=0.2,
    callbacks=cbs, verbose=1,
)

# ── 6. Evaluation ──────────────────────────────────────────────────────────────
probs      = model.predict(X_test)
y_pred     = np.argmax(probs, axis=1)
accuracy   = float(np.mean(y_pred == y_test))
confidence = np.max(probs, axis=1)

print("\n" + "═" * 60)
print("  📊 FINAL RESULTS")
print("═" * 60)
print(f"Overall Accuracy : {accuracy*100:.2f}%")
print(f"Mean Confidence  : {confidence.mean():.4f}")
report = classification_report(
    y_test, y_pred, target_names=list(STATUS_NAMES.values()), zero_division=0
)
print("\n" + report)

# Save text results
for dest in [BASE_DIR, RESULTS_DIR]:
    with open(os.path.join(dest, "classification_report.txt"), "w") as f:
        f.write(
            "HYGEIA GLOVE - AI MEDICAL SYSTEM\n"
            + "=" * 60 + "\n\nCLASSIFICATION REPORT:\n\n"
            + report
            + f"\nOverall Accuracy: {accuracy*100:.2f}%\n"
            + f"Mean Confidence: {confidence.mean():.4f}\n"
            + f"Confidence Std: {confidence.std():.4f}\n"
        )
    with open(os.path.join(dest, "final_metrics_summary.txt"), "w") as f:
        f.write(
            "HYGEIA GLOVE - FINAL RESULTS\n"
            + "=" * 60 + "\n\n"
            + f"Overall Accuracy: {accuracy*100:.2f}%\n"
            + f"Total Test Samples: {len(y_test)}\n"
            + f"Correct Predictions: {int(accuracy * len(y_test))}\n"
            + f"Incorrect Predictions: {len(y_test) - int(accuracy * len(y_test))}\n\n"
            + f"Mean Confidence: {confidence.mean():.4f}\n"
            + f"Confidence Std: {confidence.std():.4f}\n"
        )

# ── 7. Plots ───────────────────────────────────────────────────────────────────
# Training curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].plot(history.history['accuracy'],     label='Train', color='#00C9FF')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='#FF6B6B')
axes[0].set_title('Model Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].plot(history.history['loss'],     label='Train', color='#00C9FF')
axes[1].plot(history.history['val_loss'], label='Val',   color='#FF6B6B')
axes[1].set_title('Model Loss'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(alpha=.3)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "advanced_training_metrics.png"), dpi=150)
plt.close(fig)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(STATUS_NAMES.values()),
            yticklabels=list(STATUS_NAMES.values()), ax=ax)
ax.set_title('Confusion Matrix – HYGEIA AI')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "comprehensive_evaluation_heatmaps.png"), dpi=150)
plt.close(fig)

print(f"\n✅ Model  → {MODEL_OUT}")
print(f"✅ Scaler → {SCALER_OUT}")
print(f"✅ Plots  → {RESULTS_DIR}")
print("\n▶  Now run: python server.py")
