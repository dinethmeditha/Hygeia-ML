"""
HYGEIA AI – Model Training Script
===================================
This script trains the Conv1D + Bidirectional LSTM health classification model
on real patient vital sign readings (data/patient_vitals.csv).

After training completes, it saves:
  - models/hygeia_ai_model_final.h5   (the trained Keras model)
  - models/hygeia_scaler.pkl          (the fitted RobustScaler used at inference time)

Training plots and evaluation reports are saved to:
  - results/

Usage:
    python hygeia_local.py

Note:
    Only run this script once (or when you have new training data).
    The trained model is loaded by server.py at runtime.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend – safe for script use
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, Dense, Dropout,
    BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, classification_report
import joblib


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data",    "patient_vitals.csv")
MODEL_OUT   = os.path.join(BASE_DIR, "models",  "hygeia_ai_model_final.h5")
SCALER_OUT  = os.path.join(BASE_DIR, "models",  "hygeia_scaler.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "HYGEIA_AI_FINAL_RESULTS")

# Sequence length – how many consecutive readings the model sees at once
SEQ_LEN = 5

# Class labels
STATUS_NAMES = {0: "Normal", 1: "Warning", 2: "Critical"}

# Ensure output directories exist
os.makedirs(os.path.join(BASE_DIR, "models"),  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Chart style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 8)


# ==============================================================================
#  STEP 1 – LOAD DATA
# ==============================================================================

def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """
    Load the patient vitals CSV and apply physiological range validation.

    Removes any readings outside these hard bounds:
        Temperature : 30 – 45 deg C
        SpO2        :  0 – 100 %
        Pulse       :  0 – 200 BPM

    Returns a clean, reset-indexed DataFrame.
    """
    df = pd.read_csv(csv_path)

    # Coerce non-numeric entries (sensor errors) to NaN
    for col in ["Temp", "SpO2", "Pulse"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply physiological range filter
    mask = (
        df["Temp"].between(30, 45, inclusive="neither") &
        df["SpO2"].between(0, 100, inclusive="left") &
        df["Pulse"].between(0, 200, inclusive="neither")
    )
    df = df[mask].dropna().reset_index(drop=True)
    return df


# ==============================================================================
#  STEP 2 – FEATURE ENGINEERING & LABELLING
# ==============================================================================

def assign_label(row) -> int:
    """
    Rule-based labelling used to generate ground-truth labels for training.

    Classes:
        2 – Critical : values indicating immediate medical concern
        1 – Warning  : values slightly outside the healthy range
        0 – Normal   : all vitals within expected healthy range
    """
    t, s, p = row["Temp"], row["SpO2"], row["Pulse"]

    if s < 90 or t > 40.0 or t < 35.0 or p > 150 or p < 40:
        return 2  # Critical

    if s < 95 or t > 38.5 or t < 36.0 or p > 120 or p < 55:
        return 1  # Warning

    return 0  # Normal


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the 3 raw vital sign columns into 6 engineered features.

    Added features amplify compound risk signals that the model can learn:
      - Temp_SpO2_ratio  : captures combined temperature-oxygen stress
      - Pulse_SpO2_ratio : captures cardiovascular-oxygen stress
      - Health_Index     : a composite clinical quality score
    """
    df = df.copy()
    df["Temp_SpO2_ratio"]  = df["Temp"]  / (df["SpO2"]  + 1e-6)
    df["Pulse_SpO2_ratio"] = df["Pulse"] / (df["SpO2"]  + 1e-6)
    df["Health_Index"] = (
        df["SpO2"] * 0.5
        - abs(df["Temp"]  - 37.0) * 10
        - abs(df["Pulse"] - 75.0) * 0.1
    )
    return df


# ==============================================================================
#  STEP 3 – SEQUENCE CREATION (SLIDING WINDOW)
# ==============================================================================

def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Create overlapping time-series sequences from flat feature arrays.

    Instead of classifying one reading at a time, each sample contains
    `seq_len` consecutive readings (a 25-second window of history).
    The label assigned is that of the final reading in the window.

    Returns:
        X_seq : shape (N - seq_len + 1, seq_len, n_features)
        y_seq : shape (N - seq_len + 1,)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    return np.array(X_seq), np.array(y_seq)


# ==============================================================================
#  STEP 4 – MODEL ARCHITECTURE
# ==============================================================================

def build_model(input_shape: tuple, n_classes: int = 3) -> Model:
    """
    Build the HYGEIA AI deep learning model.

    Architecture:
        Conv1D  (64)  → BatchNorm → Dropout(0.3)
        Conv1D  (128) → BatchNorm → Dropout(0.3)
        BiLSTM  (64)  → Dropout(0.3)
        BiLSTM  (32)  → Dropout(0.3)
        Dense   (128) → BatchNorm → Dropout(0.4)
        Dense   (64)  → Dropout(0.3)
        Dense   (3, softmax)

    The Conv1D layers act as local pattern detectors (e.g., a sharp SpO2 drop
    over 3 steps). The Bidirectional LSTM layers then capture the temporal
    context in both directions. Softmax gives calibrated class probabilities.
    """
    inputs = Input(shape=input_shape, name="vital_signs")

    x = Conv1D(64,  kernel_size=1, activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, kernel_size=1, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64,  activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(n_classes, activation="softmax", name="status")(x)
    return Model(inputs, outputs, name="HYGEIA_AI")


# ==============================================================================
#  STEP 5 – TRAINING
# ==============================================================================

def train_model(model, X_train, y_train):
    """
    Train the model using early stopping and learning rate reduction.

    Callbacks:
        EarlyStopping       – stops training if val_loss stops improving
        ReduceLROnPlateau   – halves the learning rate on plateaus
        ModelCheckpoint     – saves the best val_accuracy checkpoint
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=20,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=10, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            MODEL_OUT, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ==============================================================================
#  STEP 6 – EVALUATION & REPORTS
# ==============================================================================

def evaluate_and_save(model, X_test, y_test):
    """
    Run model on the test set and save classification report + metrics summary.
    """
    probs      = model.predict(X_test, verbose=0)
    y_pred     = np.argmax(probs, axis=1)
    accuracy   = float(np.mean(y_pred == y_test))
    confidence = np.max(probs, axis=1)

    report = classification_report(
        y_test, y_pred,
        labels=[0, 1, 2],
        target_names=list(STATUS_NAMES.values()),
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("  FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"Mean Confidence  : {confidence.mean():.4f}")
    print("\n" + report)

    # Save classification report
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("HYGEIA AI – Patient Health Monitoring System\n")
        f.write("=" * 60 + "\n\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 60 + "\n\n")
        f.write(report)
        f.write(f"\nOverall Accuracy : {accuracy * 100:.2f}%\n")
        f.write(f"Mean Confidence  : {confidence.mean():.4f}\n")
        f.write(f"Confidence Std   : {confidence.std():.4f}\n")

    # Save metrics summary
    summary_path = os.path.join(RESULTS_DIR, "final_metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("HYGEIA AI – Final Training Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy     : {accuracy * 100:.2f}%\n")
        f.write(f"Total Test Samples   : {len(y_test)}\n")
        f.write(f"Correct Predictions  : {int(accuracy * len(y_test))}\n")
        f.write(f"Incorrect Predictions: {len(y_test) - int(accuracy * len(y_test))}\n\n")
        f.write(f"Mean Confidence  : {confidence.mean():.4f}\n")
        f.write(f"Confidence Std   : {confidence.std():.4f}\n")

    return y_pred, accuracy, confidence


# ==============================================================================
#  STEP 7 – PLOTS
# ==============================================================================

def save_training_plots(history):
    """Save training accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("HYGEIA AI – Training Metrics", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"],     label="Train", color="#00C9FF", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation", color="#FF6B6B", linewidth=2)
    axes[0].set_title("Model Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train", color="#00C9FF", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation", color="#FF6B6B", linewidth=2)
    axes[1].set_title("Model Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "advanced_training_metrics.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training plot -> {out}")


def save_confusion_matrix(y_test, y_pred):
    """Save the confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=list(STATUS_NAMES.values()),
        yticklabels=list(STATUS_NAMES.values()),
        ax=ax
    )
    ax.set_title("Confusion Matrix – HYGEIA AI", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "comprehensive_evaluation_heatmaps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix -> {out}")


def save_comprehensive_accuracy(y_test, y_pred, confidence):
    """Save a comprehensive accuracy and confidence analysis plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("HYGEIA AI – Comprehensive Accuracy Analysis", fontsize=14, fontweight="bold")

    # Bar chart of accuracy per class
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    
    # Avoid division by zero if a class has no true samples
    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc = np.nan_to_num(cm.diagonal() / cm.sum(axis=1), nan=0.0)
    
    sns.barplot(x=list(STATUS_NAMES.values()), y=class_acc, ax=axes[0], palette="viridis")
    axes[0].set_title("Accuracy per Class")
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(class_acc):
        axes[0].text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')

    # Histogram of confidence scores
    sns.histplot(confidence, bins=20, ax=axes[1], color="#00C9FF", kde=True)
    axes[1].set_title("Distribution of Prediction Confidence")
    axes[1].set_xlabel("Confidence Score")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "comprehensive_accuracy_analysis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved accuracy analysis -> {out}")


def save_time_series_panel(df):
    """Save a time series panel of the raw data."""
    # Plot a small subset (e.g. first 200 readings) to make it readable
    subset = df.head(200)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("HYGEIA AI – Patient Vitals Time Series (Sample)", fontsize=16, fontweight="bold")

    axes[0].plot(subset["Temp"], color="#FF6B6B", linewidth=2)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(subset["SpO2"], color="#00C9FF", linewidth=2)
    axes[1].set_ylabel("SpO2 (%)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(subset["Pulse"], color="#FFB86C", linewidth=2)
    axes[2].set_ylabel("Pulse (BPM)")
    axes[2].set_xlabel("Time (Reading Index)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "time_series_panel.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved time series panel -> {out}")


def save_3d_vital_signs(df):
    """Save a 3D scatter plot of the vitals colored by status."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle("HYGEIA AI – 3D Vital Signs Clustering", fontsize=14, fontweight="bold")

    # Map status int to color
    colors = {0: "green", 1: "orange", 2: "red"}
    c = df["Status"].map(colors)

    scatter = ax.scatter(df["Temp"], df["SpO2"], df["Pulse"], c=c, alpha=0.6, s=20)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("SpO2 (%)")
    ax.set_zlabel("Pulse (BPM)")

    # Custom legend
    import matplotlib.lines as mlines
    l1 = mlines.Line2D([], [], color="green", marker="o", linestyle="None", label="Normal")
    l2 = mlines.Line2D([], [], color="orange", marker="o", linestyle="None", label="Warning")
    l3 = mlines.Line2D([], [], color="red", marker="o", linestyle="None", label="Critical")
    ax.legend(handles=[l1, l2, l3], loc="upper left")

    out = os.path.join(RESULTS_DIR, "3d_vital_signs.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 3D plot -> {out}")


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print()
    print("=" * 60)
    print("  HYGEIA AI – Model Training")
    print(f"  TensorFlow : {tf.__version__}")
    gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
    print(f"  GPU        : {'Available' if gpu_available else 'Not available (using CPU)'}")
    print("=" * 60)

    # --- Load & validate ---
    print("\n[1/7] Loading patient vitals data...")
    df = load_and_validate_data(DATA_PATH)
    print(f"      {len(df)} valid readings loaded from: {DATA_PATH}")
    print(f"      Temperature range : {df['Temp'].min():.1f} - {df['Temp'].max():.1f} deg C")
    print(f"      SpO2 range        : {df['SpO2'].min():.0f}% - {df['SpO2'].max():.0f}%")
    print(f"      Pulse range       : {df['Pulse'].min():.0f} - {df['Pulse'].max():.0f} BPM")

    # --- Label & engineer features ---
    print("\n[2/7] Labelling and engineering features...")
    df["Status"] = df.apply(assign_label, axis=1)
    df = engineer_features(df)

    print("      Health status distribution:")
    for k, v in STATUS_NAMES.items():
        n = int((df["Status"] == k).sum())
        print(f"        {v:8s} : {n:4d} readings ({100 * n / len(df):.1f}%)")

    # --- Scale ---
    print("\n[3/7] Scaling features with RobustScaler...")
    feature_cols = ["Temp", "SpO2", "Pulse", "Temp_SpO2_ratio", "Pulse_SpO2_ratio", "Health_Index"]
    X = df[feature_cols].values
    y = df["Status"].values

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_OUT)
    print(f"      Scaler saved -> {SCALER_OUT}")

    # --- Create sequences ---
    print(f"\n[4/7] Creating sliding window sequences (window = {SEQ_LEN} readings)...")
    X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)
    print(f"      Sequence shape : {X_seq.shape}  (samples x timesteps x features)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    print(f"      Train samples  : {X_train.shape[0]}")
    print(f"      Test  samples  : {X_test.shape[0]}")

    # --- Build model ---
    print("\n[5/7] Building model architecture...")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # --- Train ---
    print("\n[6/7] Training the model (up to 150 epochs with early stopping)...")
    history = train_model(model, X_train, y_train)

    # --- Evaluate ---
    print("\n[7/7] Evaluating and saving results...")
    y_pred, accuracy, confidence = evaluate_and_save(model, X_test, y_test)
    
    print("      Generating plots...")
    save_training_plots(history)
    save_confusion_matrix(y_test, y_pred)
    save_comprehensive_accuracy(y_test, y_pred, confidence)
    save_time_series_panel(df)
    save_3d_vital_signs(df)

    print()
    print("=" * 60)
    print("  Training complete!")
    print(f"  Model  -> {MODEL_OUT}")
    print(f"  Scaler -> {SCALER_OUT}")
    print(f"  Results-> {RESULTS_DIR}")
    print()
    print("  Next step: python server.py")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
