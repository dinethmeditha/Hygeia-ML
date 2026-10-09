"""
HYGEIA AI – Dummy Data Generator
==================================
Generates a synthetic patient_vitals.csv file containing realistic vital sign
readings across all three health status classes: Normal, Warning, and Critical.

This is useful for:
  - Testing the model training pipeline with a fresh, balanced dataset
  - Verifying the data validation and feature engineering steps
  - Demonstrating the system without access to real patient data

The generated data uses Gaussian distributions centred around clinically
realistic values for each class, ensuring physiologically plausible readings.

Usage:
    python testing/generate_dummy_data.py

Output:
    testing/dummy_patient_vitals.csv  (synthetic dataset, not for training)
    testing/dummy_vitals_summary.txt  (summary statistics for verification)
"""

import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

OUTPUT_CSV  = os.path.join(os.path.dirname(__file__), "dummy_patient_vitals.csv")
OUTPUT_TXT  = os.path.join(os.path.dirname(__file__), "dummy_vitals_summary.txt")

# Number of rows to generate per class
N_PER_CLASS = 100


# ==============================================================================
#  GENERATION PARAMETERS
#  Mean (mu) and standard deviation (sigma) for each vital sign per class.
#  Values are based on standard clinical reference ranges.
# ==============================================================================

CLASS_PARAMS = {
    "Normal": {
        "temp":  {"mu": 36.9,  "sigma": 0.25},
        "spo2":  {"mu": 98.0,  "sigma": 1.0},
        "pulse": {"mu": 78.0,  "sigma": 8.0},
    },
    "Warning": {
        "temp":  {"mu": 38.2,  "sigma": 0.4},
        "spo2":  {"mu": 93.0,  "sigma": 1.5},
        "pulse": {"mu": 108.0, "sigma": 8.0},
    },
    "Critical": {
        "temp":  {"mu": 40.5,  "sigma": 0.5},
        "spo2":  {"mu": 87.0,  "sigma": 2.0},
        "pulse": {"mu": 135.0, "sigma": 12.0},
    },
}

# Hard physiological bounds (same as training validation)
BOUNDS = {
    "temp":  (30.0, 44.9),
    "spo2":  (50.0, 100.0),
    "pulse": (30.0, 199.0),
}


def generate_vital(mu: float, sigma: float, lo: float, hi: float) -> float:
    """Generate a single vital sign value clamped to physiological bounds."""
    value = np.random.normal(mu, sigma)
    return float(np.clip(value, lo, hi))


def generate_class_data(class_name: str, params: dict, n: int) -> pd.DataFrame:
    """Generate `n` rows of synthetic vital signs for one health class."""
    rows = []
    p = params[class_name]
    for _ in range(n):
        temp  = generate_vital(p["temp"]["mu"],  p["temp"]["sigma"],  *BOUNDS["temp"])
        spo2  = generate_vital(p["spo2"]["mu"],  p["spo2"]["sigma"],  *BOUNDS["spo2"])
        pulse = generate_vital(p["pulse"]["mu"], p["pulse"]["sigma"], *BOUNDS["pulse"])

        rows.append({
            "Temp":   round(temp, 1),
            "SpO2":   int(round(spo2)),
            "Pulse":  int(round(pulse)),
            "Status": class_name,
        })
    return pd.DataFrame(rows)


def main():
    print()
    print("=" * 55)
    print("  HYGEIA AI – Dummy Data Generator")
    print("=" * 55)
    print(f"\nGenerating {N_PER_CLASS} rows per class ({N_PER_CLASS * 3} total)...")

    frames = []
    for class_name in CLASS_PARAMS:
        df_class = generate_class_data(class_name, CLASS_PARAMS, N_PER_CLASS)
        frames.append(df_class)
        print(f"  {class_name:8s} : {len(df_class)} rows generated")

    df = pd.concat(frames, ignore_index=True)

    # Shuffle so classes are not grouped in sequence
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")

    # Save summary statistics
    lines = [
        "HYGEIA AI – Dummy Data Generator Summary",
        "=" * 55,
        f"\nTotal rows generated : {len(df)}",
        f"Rows per class       : {N_PER_CLASS}",
        "\nClass distribution:",
    ]
    for class_name, group in df.groupby("Status"):
        pct = 100 * len(group) / len(df)
        lines.append(f"  {class_name:8s} : {len(group)} rows ({pct:.1f}%)")

    lines.append("\nDescriptive statistics per class:\n")
    for class_name, group in df.groupby("Status"):
        lines.append(f"  [{class_name}]")
        for col in ["Temp", "SpO2", "Pulse"]:
            lines.append(
                f"    {col:5s}: mean={group[col].mean():.2f}, "
                f"std={group[col].std():.2f}, "
                f"min={group[col].min():.1f}, "
                f"max={group[col].max():.1f}"
            )
        lines.append("")

    summary_text = "\n".join(lines)
    with open(OUTPUT_TXT, "w") as f:
        f.write(summary_text)

    print("\nSummary Statistics:")
    print("-" * 55)
    for class_name, group in df.groupby("Status"):
        print(f"  [{class_name}]")
        for col in ["Temp", "SpO2", "Pulse"]:
            print(
                f"    {col:5s}: mean={group[col].mean():.2f}, "
                f"std={group[col].std():.2f}"
            )

    print(f"\nSummary saved to: {OUTPUT_TXT}")
    print("\nDone. You can inspect the generated CSV to verify the data distribution.")
    print("Note: This synthetic data is for testing only, not for actual model training.")
    print()


if __name__ == "__main__":
    main()
