import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.streaming.interfaces import IOnlineFeatureStore
from src.utils import get_config, setup_logger, PROJECT_ROOT

logger = setup_logger("streaming.online_store")


class FeatureSchemaMismatchError(ValueError):
    """Raised when the generated feature vector does not match FeatureSet_B schema constraints."""
    pass


class OnlineFeatureStore(IOnlineFeatureStore):
    """The single source of truth for the model input vector, aligning all features to FeatureSet_B."""

    def __init__(self, models_dir: Optional[str] = None) -> None:
        """Initializes the OnlineFeatureStore and loads the target schema.

        Args:
            models_dir: Optional directory containing preprocessing configs.
        """
        config = get_config()
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / config["paths"]["models_dir"]
        
        # Load selected features (FeatureSet_B)
        selected_features_path = self.models_dir / "selected_features.json"
        if not selected_features_path.exists():
            raise FileNotFoundError(f"Selected features not found at: {selected_features_path}")
        with open(selected_features_path, "r", encoding="utf-8") as f:
            feat_data = json.load(f)
        self.selected_features = feat_data.get("FeatureSet_B", [])
        
        # Load pipeline configuration for base categorical/boolean column lists
        pipeline_config_path = self.models_dir / "pipeline_config.json"
        if not pipeline_config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found at: {pipeline_config_path}")
        with open(pipeline_config_path, "r", encoding="utf-8") as f:
            pipe_config = json.load(f)
        self.categorical_columns = pipe_config.get("categorical_columns", [])
        self.boolean_columns = pipe_config.get("boolean_columns", [])
        
        # Store for debugging / telemetry snapshots
        self._latest_vector: Dict[str, Any] = {}
        
        logger.info("OnlineFeatureStore initialized with %d features from FeatureSet_B.", len(self.selected_features))

    def serve_feature_vector(
        self,
        telemetry: Dict[str, Any],
        rolling_stats: Dict[str, float],
        derived_features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Aligns raw telemetry, rolling statistics, and derived safety features into FeatureSet_B.

        Args:
            telemetry: Raw normalized telemetry sample.
            rolling_stats: Dictionary of mathematical rolling statistics.
            derived_features: Dictionary of calculated safety/risk indices.

        Returns:
            An immutable copy of the aligned FeatureSet_B dictionary.
        """
        # 1. Merge all input features into a single flat dict
        merged = {}
        merged.update(telemetry)
        merged.update(rolling_stats)
        merged.update(derived_features)

        vector = {}
        
        # 2. Build the exact FeatureSet_B output vector
        for feature in self.selected_features:
            # Case A: Direct match in the merged dictionary
            if feature in merged:
                val = merged[feature]
                # Convert boolean values to 1.0/0.0
                if isinstance(val, bool):
                    vector[feature] = 1.0 if val else 0.0
                else:
                    vector[feature] = float(val)
                continue

            # Case B: One-hot encoded categorical variable (e.g. Permit_Type_None, Shift_Type_Night)
            encoded = False
            for base_col in self.categorical_columns:
                prefix = f"{base_col}_"
                if feature.startswith(prefix):
                    category_val = feature[len(prefix):]
                    raw_val = telemetry.get(base_col)
                    # Check matching string category value
                    if raw_val is not None and str(raw_val) == category_val:
                        vector[feature] = 1.0
                    else:
                        vector[feature] = 0.0
                    encoded = True
                    break
            
            if encoded:
                continue

            # Case C: One-hot encoded boolean variable (e.g. PPE_Compliance_True, PPE_Compliance_False)
            for base_col in self.boolean_columns:
                prefix = f"{base_col}_"
                if feature.startswith(prefix):
                    bool_suffix = feature[len(prefix):]
                    raw_val = telemetry.get(base_col)
                    
                    if raw_val is not None:
                        # Standardize input truthiness
                        raw_bool = bool(raw_val)
                        if bool_suffix == "True" and raw_bool:
                            vector[feature] = 1.0
                        elif bool_suffix == "False" and not raw_bool:
                            vector[feature] = 1.0
                        else:
                            vector[feature] = 0.0
                    else:
                        vector[feature] = 0.0
                    encoded = True
                    break

            if encoded:
                continue

            # Case D: Missing feature fallback (defaults to 0.0)
            logger.debug("Feature %s not found in inputs, defaulting to 0.0", feature)
            vector[feature] = 0.0

        # 3. Perform final strict feature quality and schema validation checks
        if len(vector) != len(self.selected_features):
            raise FeatureSchemaMismatchError(f"Feature count mismatch: expected {len(self.selected_features)}, generated {len(vector)}")

        # Validate feature ordering
        for idx, (k, _) in enumerate(vector.items()):
            expected_key = self.selected_features[idx]
            if k != expected_key:
                raise FeatureSchemaMismatchError(f"Feature ordering mismatch at position {idx}: expected {expected_key}, found {k}")

        # Validate types and check for NaN / Inf
        for k, v in vector.items():
            if not isinstance(v, float):
                raise FeatureSchemaMismatchError(f"Feature type invalid for {k}: expected float, found {type(v).__name__}")
            if math.isnan(v):
                raise FeatureSchemaMismatchError(f"NaN value detected in feature vector for column: {k}")
            if math.isinf(v):
                raise FeatureSchemaMismatchError(f"Infinite value detected in feature vector for column: {k}")

        self._latest_vector = vector
        return vector.copy()

    def get_latest_vector(self) -> Dict[str, Any]:
        """Returns the latest assembled feature vector snapshot."""
        return self._latest_vector.copy()

    def get_schema(self) -> List[str]:
        """Returns the list of target features in the schema.

        Returns:
            List of feature names in the FeatureSet_B schema.
        """
        return list(self.selected_features)
