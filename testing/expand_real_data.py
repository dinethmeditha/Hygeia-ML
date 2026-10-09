import os
import random
import pandas as pd
import numpy as np

# Set seed for reproducibility
random.seed(101)
np.random.seed(101)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "patient_vitals.csv")

def generate_realistic_data(target_rows=500):
    # Base distributions for different patient states to make it look real
    
    states = [
        # Normal (majority)
        {"temp_mu": 36.8, "temp_sigma": 0.3, "spo2_mu": 98, "spo2_sigma": 1.5, "pulse_mu": 76, "pulse_sigma": 8, "weight": 0.65},
        # Mild Fever / Warning
        {"temp_mu": 38.3, "temp_sigma": 0.4, "spo2_mu": 94, "spo2_sigma": 1.5, "pulse_mu": 105, "pulse_sigma": 10, "weight": 0.15},
        # Low Oxygen / Warning
        {"temp_mu": 37.0, "temp_sigma": 0.3, "spo2_mu": 92, "spo2_sigma": 1.5, "pulse_mu": 115, "pulse_sigma": 10, "weight": 0.10},
        # Critical state (High fever + tachycardia)
        {"temp_mu": 40.5, "temp_sigma": 0.5, "spo2_mu": 96, "spo2_sigma": 1.5, "pulse_mu": 140, "pulse_sigma": 15, "weight": 0.05},
        # Critical state (Severe Hypoxaemia)
        {"temp_mu": 36.9, "temp_sigma": 0.4, "spo2_mu": 85, "spo2_sigma": 3.0, "pulse_mu": 120, "pulse_sigma": 20, "weight": 0.05},
    ]
    
    rows = []
    
    for _ in range(target_rows):
        # Pick a state based on weights
        state = random.choices(states, weights=[s["weight"] for s in states])[0]
        
        t = np.random.normal(state["temp_mu"], state["temp_sigma"])
        s = np.random.normal(state["spo2_mu"], state["spo2_sigma"])
        p = np.random.normal(state["pulse_mu"], state["pulse_sigma"])
        
        # Format them nicely
        t_str = f"{t:.1f}"
        s_str = f"{min(100, int(round(s)))}"
        p_str = f"{int(round(p))}"
        
        # Introduce "small errors" and sensor glitches (like real world data)
        err_chance = random.random()
        if err_chance < 0.02:   # 2% chance of sensor disconnect (empty or weird value)
            col_to_break = random.choice(["temp", "spo2", "pulse"])
            if col_to_break == "temp":
                t_str = random.choice(["", "NaN", "err", "0.0"])
            elif col_to_break == "spo2":
                s_str = random.choice(["", "NaN", "-", "0"])
            else:
                p_str = random.choice(["", "NaN", "error", "-1"])
        elif err_chance < 0.04: # 2% chance of physiologically impossible outlier (sensor artifact)
            col_to_break = random.choice(["temp", "spo2", "pulse"])
            if col_to_break == "temp":
                t_str = f"{random.uniform(15.0, 25.0):.1f}" # Room temp
            elif col_to_break == "spo2":
                s_str = "150" # Impossible SpO2
            else:
                p_str = "250" # Extreme pulse
                
        rows.append({"Temp": t_str, "SpO2": s_str, "Pulse": p_str})
        
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Successfully generated {target_rows} rows of realistic data to {CSV_PATH}")

if __name__ == "__main__":
    generate_realistic_data(500) # 500 rows + 1 header = 501 lines
