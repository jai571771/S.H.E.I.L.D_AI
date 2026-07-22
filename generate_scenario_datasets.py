import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def main():
    print("Generating scenario datasets for S.H.E.I.L.D. hackathon demonstration...")
    
    # Load base dataset
    csv_path = "test_model.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/test_model.csv"
    if not os.path.exists(csv_path):
        print(f"Error: test_model.csv not found at root or data/. Cannot generate scenarios.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded base dataset from {csv_path}. Shape: {df.shape}")

    # Create target directory
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Group rows by risk class
    df_low = df[df["Future_Risk_Level"] == "Low"].reset_index(drop=True)
    df_med = df[df["Future_Risk_Level"] == "Medium"].reset_index(drop=True)
    df_high = df[df["Future_Risk_Level"] == "High"].reset_index(drop=True)
    df_crit = df[df["Future_Risk_Level"] == "Critical"].reset_index(drop=True)

    # Helper to generate timestamps
    start_time = datetime(2026, 10, 1, 9, 0, 0)
    def make_timestamps(n):
        return [(start_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(n)]

    # 🟢 Scenario 1: Normal Operation (Low Risk)
    print("Generating Scenario 1: Normal Operation...")
    s1_rows = df_low.sample(n=50, replace=True, random_state=42).copy()
    s1_rows["Timestamp"] = make_timestamps(50)
    s1_rows["Future_Risk_Level"] = "Low"
    s1_rows.to_csv(os.path.join(output_dir, "normal_operation.csv"), index=False)

    # 🟡 Scenario 2: Gas Pressure Rise (Medium Risk)
    print("Generating Scenario 2: Gas Pressure Rise...")
    s2_base = df_low.sample(n=15, replace=True, random_state=42).copy()
    s2_event = df_med.sample(n=35, replace=True, random_state=42).copy()
    s2_rows = pd.concat([s2_base, s2_event]).reset_index(drop=True)
    s2_rows["Timestamp"] = make_timestamps(50)
    
    # Apply pressure rise ramp
    for i in range(15, 50):
        factor = (i - 15) / 35.0
        s2_rows.loc[i, "BF_Gas_Pressure"] = 2.0 + (factor * 2.2) # Ramps up to 4.2 bar (Warn threshold exceeded)
        s2_rows.loc[i, "CO_Gas_Pressure"] = 2.2 + (factor * 2.0) # Ramps up to 4.2 bar
        s2_rows.loc[i, "BF_CO"] = 1.2 + (factor * 1.5) # Ramps up to 2.7%
        s2_rows.loc[i, "CO_CO"] = 1.0 + (factor * 1.6) # Ramps up to 2.6%
    s2_rows.to_csv(os.path.join(output_dir, "gas_pressure_rise.csv"), index=False)

    # 🟠 Scenario 3: Cooling System Failure (High Risk)
    print("Generating Scenario 3: Cooling System Failure...")
    s3_base = df_low.sample(n=15, replace=True, random_state=43).copy()
    s3_event = df_high.sample(n=35, replace=True, random_state=43).copy()
    s3_rows = pd.concat([s3_base, s3_event]).reset_index(drop=True)
    s3_rows["Timestamp"] = make_timestamps(50)
    
    # Apply cooling failure ramp
    for i in range(15, 50):
        factor = (i - 15) / 35.0
        s3_rows.loc[i, "BF_Cooling_Water_Flow"] = 150.0 - (factor * 70.0) # Drops to 80.0 L/min
        s3_rows.loc[i, "BF_Top_Temperature"] = 200.0 + (factor * 110.0) # Rises to 310°C
        s3_rows.loc[i, "BF_Cooling_Water_Temperature"] = 35.0 + (factor * 20.0) # Rises to 55°C
    s3_rows.to_csv(os.path.join(output_dir, "cooling_failure.csv"), index=False)

    # 🔴 Scenario 4: Compound Hazard (Critical Risk)
    print("Generating Scenario 4: Compound Hazard...")
    s4_base = df_low.sample(n=10, replace=True, random_state=44).copy()
    s4_med = df_med.sample(n=10, replace=True, random_state=44).copy()
    s4_high = df_high.sample(n=10, replace=True, random_state=44).copy()
    s4_crit = df_crit.sample(n=20, replace=True, random_state=44).copy()
    s4_rows = pd.concat([s4_base, s4_med, s4_high, s4_crit]).reset_index(drop=True)
    s4_rows["Timestamp"] = make_timestamps(50)
    
    # Apply compounding disaster ramps
    for i in range(10, 50):
        factor = (i - 10) / 40.0
        s4_rows.loc[i, "BF_Gas_Pressure"] = 2.0 + (factor * 2.8) # Ramps to 4.8 bar
        s4_rows.loc[i, "BF_Cooling_Water_Flow"] = 150.0 - (factor * 95.0) # Drops to 55 L/min
        s4_rows.loc[i, "BF_Top_Temperature"] = 200.0 + (factor * 180.0) # Rises to 380°C
        s4_rows.loc[i, "BF_CO"] = 1.2 + (factor * 3.6) # Ramps to 4.8%
        s4_rows.loc[i, "CO_CO"] = 1.0 + (factor * 3.2) # Ramps to 4.2%
        s4_rows.loc[i, "BF_Blower_Vibration"] = 1.5 + (factor * 3.0) # Ramps to 4.5 mm/s
    s4_rows.to_csv(os.path.join(output_dir, "compound_hazard.csv"), index=False)

    # 🟣 Scenario 5: Maintenance Mode (Low/Medium Risk)
    print("Generating Scenario 5: Maintenance Mode...")
    s5_rows = df_low.sample(n=50, replace=True, random_state=45).copy()
    s5_rows["Timestamp"] = make_timestamps(50)
    s5_rows["Maintenance_Active"] = True
    s5_rows["Permit_Type"] = "Hot Work"
    s5_rows["PPE_Compliance"] = True
    s5_rows["Worker_Count"] = 6
    s5_rows["Gas_Test_Completed"] = True
    # Minor stable emissions
    s5_rows["BF_CO"] = 1.8
    s5_rows["CO_CO"] = 1.6
    s5_rows.to_csv(os.path.join(output_dir, "maintenance_mode.csv"), index=False)

    # 🟤 Scenario 6: Equipment Wear (Medium Risk)
    print("Generating Scenario 6: Equipment Wear...")
    s6_rows = df_low.sample(n=50, replace=True, random_state=46).copy()
    s6_rows["Timestamp"] = make_timestamps(50)
    
    # Mechanical wear ramp
    for i in range(50):
        factor = i / 50.0
        s6_rows.loc[i, "BF_Blower_Vibration"] = 1.2 + (factor * 2.4) # Ramps to 3.6 mm/s
        s6_rows.loc[i, "CO_Pusher_Vibration"] = 1.4 + (factor * 2.0) # Ramps to 3.4 mm/s
        s6_rows.loc[i, "BF_Blower_Current"] = 3000 + (factor * 1500) # Current rising due to load
    s6_rows.to_csv(os.path.join(output_dir, "equipment_wear.csv"), index=False)

    # ⚫ Scenario 7: Emergency Shutdown (Critical Risk)
    print("Generating Scenario 7: Emergency Shutdown...")
    s7_base = df_crit.sample(n=10, replace=True, random_state=47).copy()
    s7_event = df_crit.sample(n=40, replace=True, random_state=47).copy()
    s7_rows = pd.concat([s7_base, s7_event]).reset_index(drop=True)
    s7_rows["Timestamp"] = make_timestamps(50)
    
    # Apply emergency trip ramp (shutting down operations)
    for i in range(10, 50):
        factor = (i - 10) / 40.0
        s7_rows.loc[i, "BF_Blast_Flow"] = max(0.0, s7_rows.loc[10, "BF_Blast_Flow"] * (1.0 - factor))
        s7_rows.loc[i, "BF_Blower_Current"] = max(0.0, s7_rows.loc[10, "BF_Blower_Current"] * (1.0 - factor))
        s7_rows.loc[i, "CO_Pusher_Current"] = 0.0
        s7_rows.loc[i, "BF_Top_Temperature"] = 250.0 - (factor * 160.0) # Drops to 90°C
        s7_rows.loc[i, "CO_Oven_Temperature"] = 1000.0 - (factor * 400.0) # Drops to 600°C
        s7_rows.loc[i, "Worker_Count"] = 0
    s7_rows.to_csv(os.path.join(output_dir, "emergency_shutdown.csv"), index=False)

    print("All scenario datasets successfully generated under datasets/ folder.")

if __name__ == "__main__":
    main()
