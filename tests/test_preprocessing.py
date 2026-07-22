import unittest
import pandas as pd
import numpy as np
from src.preprocessing import PreprocessingPipeline

class TestPreprocessingPipeline(unittest.TestCase):
    """Tests the PreprocessingPipeline class."""

    def test_clean_and_transform(self) -> None:
        # Create a mock dataframe
        mock_df = pd.DataFrame([
            {
                "Timestamp": "2026-07-17 12:00:00",
                "BF_Blast_Flow": 1500.0,
                "BF_Top_Temperature": 120.0,
                "Shift_Type": "Day",
                "PPE_Compliance": "Yes",
                "Maintenance_Active": False,
                "CO_Gas_Pressure": 1.2
            },
            {
                "Timestamp": "2026-07-17 12:01:00",
                "BF_Blast_Flow": 2300.0,
                "BF_Top_Temperature": np.nan,  # testing missing value imputation
                "Shift_Type": "Night",
                "PPE_Compliance": "No",
                "Maintenance_Active": True,
                "CO_Gas_Pressure": 3.4
            },
            {
                "Timestamp": "2026-07-17 12:02:00",
                "BF_Blast_Flow": 1800.0,
                "BF_Top_Temperature": 150.0,
                "Shift_Type": "Day",
                "PPE_Compliance": "Yes",
                "Maintenance_Active": False,
                "CO_Gas_Pressure": np.nan  # testing missing value imputation
            }
        ])

        pipeline = PreprocessingPipeline()
        
        # Clean
        cleaned_df = pipeline.clean_dataset(mock_df)
        self.assertFalse(cleaned_df["CO_Gas_Pressure"].isnull().any())
        self.assertFalse(cleaned_df["BF_Top_Temperature"].isnull().any())
        
        # Fit transform
        trans_df, feature_names = pipeline.fit_transform_features(cleaned_df)
        self.assertIsNotNone(pipeline.encoder)
        self.assertIsNotNone(pipeline.scaler)
        
        # Outputs should have columns corresponding to one-hot encoding, booleans, and scaled numerics
        self.assertIn("Shift_Type_Day", trans_df.columns)
        self.assertIn("Shift_Type_Night", trans_df.columns)
        self.assertIn("Maintenance_Active", trans_df.columns)
        self.assertIn("BF_Blast_Flow", trans_df.columns)
        self.assertEqual(len(trans_df), 3)

    def test_schema_mismatch_error(self) -> None:
        pipeline = PreprocessingPipeline()
        
        # Fit on a simple df
        fit_df = pd.DataFrame([{"BF_Blast_Flow": 1500.0, "CO_Gas_Pressure": 1.2}])
        pipeline.fit_transform_features(fit_df)
        
        # Transform on a df missing CO_Gas_Pressure
        test_df = pd.DataFrame([{"BF_Blast_Flow": 1800.0}])
        
        from src.preprocessing import FeatureSchemaMismatchError
        with self.assertRaises(FeatureSchemaMismatchError) as context:
            pipeline.transform_features(test_df)
            
        self.assertIn("CO_Gas_Pressure", context.exception.missing_features)

if __name__ == "__main__":
    unittest.main()
