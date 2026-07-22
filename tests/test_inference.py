import unittest
from pathlib import Path
import joblib
from src.inference import InferenceEngine
from src.utils import PROJECT_ROOT

class TestInferenceEngine(unittest.TestCase):
    """Tests the InferenceEngine class."""

    def test_predict_single(self) -> None:
        # Check if the serialized model exists before running this test
        models_dir = PROJECT_ROOT / "models" / "preprocessing"
        model_path = models_dir / "risk_classifier.pkl"
        
        if not model_path.exists():
            self.skipTest("Certified classifier not found. Skipping InferenceEngine test.")

        engine = InferenceEngine()
        
        # Load a single row from the actual dataset to use as a realistic mock telemetry sample
        import pandas as pd
        from src.dataset_loader import DatasetLoader
        
        loader = DatasetLoader()
        df, _ = loader.load_and_validate()
        if df is None or df.empty:
            self.skipTest("Dataset not available to load mock telemetry row. Skipping.")
            
        mock_telemetry = df.iloc[0].to_dict()
        mock_telemetry.pop("Future_Risk_Level", None)
        
        result = engine.predict_single(mock_telemetry)
        
        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn("shap_contributions", result)
        self.assertIn("recommendations", result)
        
        self.assertEqual(len(result["probabilities"]), 4)

if __name__ == "__main__":
    unittest.main()
