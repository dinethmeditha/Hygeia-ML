"""
HYGEIA AI - Patient Vital Signs Data Streamer
=============================================
Streams patient_vitals.csv row-by-row to the HYGEIA server
to simulate a continuous live data feed for demonstration purposes.

Run AFTER starting server.py:   python data_streamer.py
"""

import time
import requests
import pandas as pd
import os

print("=" * 50)
print("  HYGEIA AI - Data Streamer")
print("=" * 50)
print("Reading data/patient_vitals.csv...")

csv_path = os.path.join(os.path.dirname(__file__), "data", "patient_vitals.csv")
if not os.path.exists(csv_path):
    print("Error: data/patient_vitals.csv not found!")
    exit(1)

df = pd.read_csv(csv_path).dropna()
url = "http://localhost:5000/api/reading"

print(f"Loaded {len(df)} patient records.")
print("Streaming data to server. Press Ctrl+C to stop.")
print()

while True:
    for index, row in df.iterrows():
        payload = {
            "temp":  row['Temp'],
            "spo2":  row['SpO2'],
            "pulse": row['Pulse']
        }
        try:
            response = requests.post(url, json=payload, timeout=2)
            if response.status_code == 200:
                result = response.json()
                print(f"[{index+1}/{len(df)}] T={payload['temp']}C | SpO2={payload['spo2']}% | BPM={payload['pulse']} -> {result.get('classification','?')}")
            else:
                print(f"Server error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Connection failed: {e}. Retrying next cycle...")

        time.sleep(2)
    print("[Data Streamer] CSV completed - looping back to start...")
