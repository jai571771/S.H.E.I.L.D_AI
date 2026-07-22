import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from src.utils import get_config, setup_logger, PROJECT_ROOT

logger = setup_logger("pipeline.dataset_loader")

class DatasetLoader:
    """Loads and validates the structural integrity of the industrial safety dataset."""

    def __init__(self, dataset_path: Optional[str] = None) -> None:
        """Initializes the DatasetLoader with parameters from config."""
        config = get_config()
        if dataset_path is None:
            self.dataset_path = PROJECT_ROOT / config["paths"]["dataset_path"]
        else:
            self.dataset_path = Path(dataset_path)
            
        logger.info("DatasetLoader initialized with path: %s", self.dataset_path)

    def load_and_validate(self) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Loads the dataset and validates its structure.

        Returns:
            A tuple of (loaded_dataframe, validation_metadata_dictionary).
        """
        metadata: Dict[str, Any] = {
            "success": False,
            "rows": 0,
            "columns": 0,
            "memory_mb": 0.0,
            "target_present": False,
            "error": None
        }

        if not self.dataset_path.exists():
            err_msg = f"Dataset not found at: {self.dataset_path}"
            logger.error(err_msg)
            metadata["error"] = err_msg
            return None, metadata

        try:
            logger.info("Loading dataset: %s...", self.dataset_path.name)
            df = pd.read_csv(self.dataset_path)
            
            # Compute footprint
            memory_usage_bytes = df.memory_usage(deep=True).sum()
            memory_mb = float(memory_usage_bytes) / (1024 * 1024)

            metadata["rows"] = len(df)
            metadata["columns"] = len(df.columns)
            metadata["memory_mb"] = memory_mb
            metadata["target_present"] = "Future_Risk_Level" in df.columns
            
            # Assertions / Validations
            if len(df) != 50000:
                logger.warning("Unexpected row count: %d (expected 50000)", len(df))
            if len(df.columns) != 193:
                logger.warning("Unexpected column count: %d (expected 193)", len(df.columns))
            if not metadata["target_present"]:
                logger.error("Target variable 'Future_Risk_Level' is missing!")
                metadata["error"] = "Missing target column 'Future_Risk_Level'"
                return None, metadata

            metadata["success"] = True
            logger.info(
                "Dataset successfully loaded. Shape: %s, Memory: %.2f MB, Target present: %s",
                df.shape, memory_mb, metadata["target_present"]
            )
            return df, metadata

        except Exception as e:
            err_msg = f"Failed to load dataset: {str(e)}"
            logger.exception(err_msg)
            metadata["error"] = err_msg
            return None, metadata

if __name__ == "__main__":
    loader = DatasetLoader()
    df, meta = loader.load_and_validate()
    print("Metadata:", meta)
