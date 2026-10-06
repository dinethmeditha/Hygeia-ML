import time
import requests
import pandas as pd
import os

print("Starting ESP32 Simulator...")
print("Reading data/glove_readings.csv...")

csv_path = os.path.join(os.path.dirname(__file__), "data", "glove_readings.csv")
if not os.path.exists(csv_path):
    print("Error: data/glove_readings.csv not found!")
    exit(1)

df = pd.read_csv(csv_path)

# Drop NaNs
df = df.dropna()

url = "http://localhost:5000/api/reading"

print("Starting to stream data to server. Press Ctrl+C to stop.")

while True:
    for index, row in df.iterrows():
        payload = {
            "temp": row['Temp'],
            "spo2": row['SpO2'],
            "pulse": row['Pulse']
        }
        
        try:
            response = requests.post(url, json=payload, timeout=2)
            if response.status_code == 200:
                print(f"Sent: {payload} -> {response.json().get('classification')}")
            else:
                print(f"Error: Server returned status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Connection failed: {e}. Is the server running?")
        
        # Wait 2 seconds before sending the next reading to simulate real-time
        time.sleep(2)
