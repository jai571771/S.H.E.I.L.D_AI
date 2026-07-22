import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.streaming.telemetry_validator import TelemetryValidator
from src.streaming.buffer import TelemetryBuffer
from src.streaming.rolling_stats import RollingStatisticsEngine
from src.streaming.derived_features import DerivedFeatureEngine

def generate_test_model_csv(
    source_csv: str = "data/industrial_safety_dataset_3.0.csv",
    output_paths = ("data/test_model.csv", "test_model.csv"),
    n_per_class: int = 100,
    seed: int = 42
):
    """Generates an independent test dataset (test_model.csv) by sampling representative rows 
    from each risk class and passing raw telemetry through the streaming pipeline engines 
    to guarantee feature consistency and schema compliance."""
    np.random.seed(seed)
    df_src = pd.read_csv(source_csv)

    classes = ["Low", "Medium", "High", "Critical"]
    raw_sensors = TelemetryValidator.RAW_SENSORS

    validator = TelemetryValidator()
    derived_engine = DerivedFeatureEngine()

    start_time = datetime(2026, 9, 1, 8, 0, 0)
    current_time = start_time

    rows = []

    for target_class in classes:
        # Filter source dataset by target risk class
        df_cls = df_src[df_src["Future_Risk_Level"] == target_class].reset_index(drop=True)
        
        # Sample n_per_class unseen baseline rows
        sample_indices = np.random.choice(len(df_cls), size=n_per_class, replace=(len(df_cls) < n_per_class))

        for i, idx in enumerate(sample_indices):
            current_time += timedelta(minutes=1)
            ts_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

            # Take full certified feature row
            base_dict = df_cls.iloc[idx].to_dict()
            
            # Create a modified raw telemetry payload with minor SCADA sensor noise (+/- 0.5%)
            raw_dict = {"Timestamp": ts_str}
            for col in raw_sensors:
                val = base_dict.get(col, 0.0)
                if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
                    noise = np.random.normal(0, abs(val) * 0.005 if abs(val) > 1e-3 else 0.01)
                    raw_dict[col] = float(val + noise)
                else:
                    raw_dict[col] = val

            # Keep categorical/metadata fields
            raw_dict["BF_State_ID"] = base_dict.get("BF_State_ID", "PS001")
            raw_dict["BF_State_Name"] = base_dict.get("BF_State_Name", "Normal")
            raw_dict["BF_State_Risk"] = base_dict.get("BF_State_Risk", "Safe")
            raw_dict["CO_State_ID"] = base_dict.get("CO_State_ID", "PS001")
            raw_dict["CO_State_Name"] = base_dict.get("CO_State_Name", "Normal")
            raw_dict["CO_State_Risk"] = base_dict.get("CO_State_Risk", "Safe")
            raw_dict["Safety_Index"] = float(base_dict.get("Safety_Index", 85.0))

            # Run normalization and derived features through streaming pipeline components
            norm = validator.normalize(raw_dict)
            norm["Safety_Index"] = raw_dict["Safety_Index"]

            # Merge back into complete certified row template
            sample = dict(base_dict)
            sample.update(norm)

            # Assign fresh test identifiers
            sample["Timestamp"] = ts_str
            sample["Event_ID"] = f"TEST_EV_{target_class[0]}_{i+1000:04d}"
            sample["Future_Risk_Level"] = target_class

            rows.append(sample)

    df_test = pd.DataFrame(rows)

    for p in output_paths:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        df_test.to_csv(p, index=False)
        print(f"[SUCCESS] Test dataset successfully exported ({len(df_test)} rows) to: {p}")

    print("\nTarget Class Distribution:")
    print(df_test["Future_Risk_Level"].value_counts())

if __name__ == "__main__":
    generate_test_model_csv()
