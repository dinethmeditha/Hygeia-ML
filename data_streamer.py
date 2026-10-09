"""
HYGEIA AI – Patient Vital Signs Data Streamer
=============================================
This script reads patient vital sign records from data/patient_vitals.csv
and streams them row-by-row to the HYGEIA server's /api/reading endpoint.

It simulates a continuous live data feed, which is useful for demonstrations
and functional testing without requiring the physical ESP32 hardware.

Prerequisites:
    server.py must be running first (python server.py)

Usage:
    python data_streamer.py

The streamer loops back to the beginning of the CSV after reaching the
last row, so it can run continuously for as long as needed.
"""

import os
import time
import requests
import pandas as pd


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

SERVER_URL    = "http://localhost:5000/api/reading"
CSV_PATH      = os.path.join(os.path.dirname(__file__), "data", "patient_vitals.csv")
SEND_INTERVAL = 2   # Seconds between each streamed reading


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print()
    print("=" * 50)
    print("  HYGEIA AI – Patient Data Streamer")
    print("=" * 50)

    # Verify the data file exists before starting
    if not os.path.exists(CSV_PATH):
        print(f"\nError: Training data file not found at:\n  {CSV_PATH}")
        print("Make sure data/patient_vitals.csv is present before running this script.")
        return

    df = pd.read_csv(CSV_PATH).dropna()
    total = len(df)
    print(f"\nLoaded {total} patient records from: {CSV_PATH}")
    print(f"Streaming to: {SERVER_URL}")
    print(f"Interval    : {SEND_INTERVAL} seconds per reading")
    print("\nPress Ctrl+C to stop.\n")

    cycle = 1
    while True:
        print(f"--- Cycle {cycle} (streaming {total} records) ---")
        for index, row in df.iterrows():
            payload = {
                "temp":  float(row["Temp"]),
                "spo2":  float(row["SpO2"]),
                "pulse": float(row["Pulse"]),
            }
            try:
                response = requests.post(SERVER_URL, json=payload, timeout=3)
                if response.status_code == 200:
                    result = response.json()
                    classification = result.get("classification", "?")
                    confidence     = result.get("confidence", 0.0)
                    print(
                        f"  [{index + 1:>3}/{total}] "
                        f"Temp={payload['temp']:.1f}C  "
                        f"SpO2={payload['spo2']:.0f}%  "
                        f"Pulse={payload['pulse']:.0f}bpm  "
                        f"-> {classification} ({confidence * 100:.1f}%)"
                    )
                else:
                    print(f"  [{index + 1}/{total}] Server error: HTTP {response.status_code}")

            except requests.exceptions.ConnectionError:
                print(f"  [{index + 1}/{total}] Cannot connect to server. Is server.py running?")

            except requests.exceptions.Timeout:
                print(f"  [{index + 1}/{total}] Request timed out – server may be busy.")

            time.sleep(SEND_INTERVAL)

        print(f"--- Cycle {cycle} complete. Looping back to start... ---\n")
        cycle += 1


if __name__ == "__main__":
    main()
