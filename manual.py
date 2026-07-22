import json
import pandas as pd
from src.utils import get_config, PROJECT_ROOT
from src.inference.predictor import predict

def test_custom_prediction():
    print("==================================================")
    print("   S.H.E.I.L.D. MANUAL INFERENCE TEST UTILITY     ")
    print("==================================================")

    # 1. Load configuration and dataset for a complete base row template
    config = get_config()
    dataset_path = PROJECT_ROOT / config["paths"]["dataset_path"]
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}. Please check your files.")
        return

    print(f"Loading template telemetry schema from: {dataset_path.name}")
    df_raw = pd.read_csv(dataset_path)
    
    # Extract the first row as a base template
    base_sample = df_raw.iloc[0].to_dict()
    
    # Remove ground truth risk labels if present
    base_sample.pop("Future_Risk_Level", None)

    # Convert NumPy values to standard Python types for JSON compatibility
    sample = {}
    for k, v in base_sample.items():
        if pd.isna(v):
            sample[k] = None
        elif hasattr(v, "item"):  # numpy scalars
            sample[k] = v.item()
        else:
            sample[k] = v

    # 2. Update with the user's custom test parameters
    user_updates = {
        "BF_CO": 22.5,
        "BF_H2": 2.1,
        "BF_Top_Temperature": 205,
        "BF_Gas_Pressure": 19.8,
        "CO_CO": 2.8,
        "CO_NH3": 1.2,
        "Worker_Count": 6,
        "Maintenance_Active": False,
        "PPE_Compliance": "Yes"  # Dataset expects "Yes"/"No" categorical values
    }
    sample.update(user_updates)

    print("\nEvaluating custom sample telemetry inputs:")
    for k, v in user_updates.items():
        print(f"  - {k}: {v}")

    # 3. Execute inference engine prediction
    try:
        result = predict(sample)
        print("\n================ Prediction Result ================")
        print(json.dumps(result, indent=4))
        print("===================================================")
    except Exception as e:
        print(f"\nInference failed: {str(e)}")

if __name__ == "__main__":
    test_custom_prediction()
