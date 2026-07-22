import unittest
from pathlib import Path
import pandas as pd
from src.dataset_loader import DatasetLoader
from src.utils import PROJECT_ROOT

class TestDatasetLoader(unittest.TestCase):
    """Tests the DatasetLoader class."""

    def test_load_and_validate(self) -> None:
        loader = DatasetLoader()
        df, meta = loader.load_and_validate()
        
        # If the file exists, it should load correctly. If not, it should fail gracefully with false success.
        if Path(loader.dataset_path).exists():
            self.assertTrue(meta["success"])
            self.assertIsNotNone(df)
            self.assertIn("Future_Risk_Level", df.columns)
            self.assertEqual(len(df), 50000)
            self.assertEqual(len(df.columns), 193)
        else:
            self.assertFalse(meta["success"])
            self.assertIsNone(df)
            self.assertIsNotNone(meta["error"])

if __name__ == "__main__":
    unittest.main()
