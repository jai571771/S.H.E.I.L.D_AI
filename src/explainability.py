import logging
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from src.utils import get_config, setup_logger, PROJECT_ROOT

logger = setup_logger("pipeline.explainability")

class ShapExplainerManager:
    """Manages real-time SHAP local explanations and global feature attribution plots."""

    def __init__(self, models_dir: Optional[str] = None, reports_dir: Optional[str] = None) -> None:
        """Initializes the ShapExplainerManager."""
        config = get_config()
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / config["paths"]["models_dir"]
        self.reports_dir = Path(reports_dir) if reports_dir else PROJECT_ROOT / config["paths"]["reports_dir"]
        self.plots_dir = self.reports_dir / "plots" / "test_evaluation"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.models_dir / "risk_classifier.pkl"
        self._model = None
        self._explainer = None
        self._features_list: List[str] = []

    @property
    def model(self) -> Any:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Certified classifier not found at {self.model_path}")
            self._model = joblib.load(self.model_path)
        return self._model

    @property
    def explainer(self) -> shap.TreeExplainer:
        if self._explainer is None:
            logger.info("Initializing shap.TreeExplainer on certified model...")
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    @property
    def features_list(self) -> List[str]:
        if not self._features_list:
            feat_path = self.models_dir / "selected_features.json"
            if not feat_path.exists():
                raise FileNotFoundError(f"Feature names list not found at {feat_path}")
            with open(feat_path, "r", encoding="utf-8") as f:
                feat_dict = json.load(f)
            self._features_list = feat_dict["FeatureSet_B"]
        return self._features_list

    def explain_sample(self, X_sample: pd.DataFrame, predicted_class_idx: int) -> List[Dict[str, Any]]:
        """Computes local SHAP values for a single sample and returns top contributing features.

        Args:
            X_sample: A single-row DataFrame containing features matching FeatureSet_B.
            predicted_class_idx: Index of the predicted class (0-3).

        Returns:
            A list of dicts with keys: "feature", "val", "contribution".
        """
        # Ensure correct column order
        X_sample = X_sample[self.features_list]
        
        # Compute SHAP values
        # For multi-class, TreeExplainer returns a list of arrays (one per class), or a 3D array [samples, features, classes]
        shap_values_output = self.explainer.shap_values(X_sample)
        
        # Handle shape differences between shap versions:
        # Some versions return list of len = num_classes, each of shape (num_samples, num_features)
        # Some versions return array of shape (num_samples, num_features, num_classes)
        if isinstance(shap_values_output, list):
            # List of arrays per class. Take the array for the predicted class
            shap_values_for_class = shap_values_output[predicted_class_idx][0]
        elif isinstance(shap_values_output, np.ndarray):
            if len(shap_values_output.shape) == 3:
                # Shape: (samples, features, classes)
                shap_values_for_class = shap_values_output[0, :, predicted_class_idx]
            elif len(shap_values_output.shape) == 2:
                # Binary/Regression case or single class output
                shap_values_for_class = shap_values_output[0]
            else:
                shap_values_for_class = shap_values_output
        else:
            shap_values_for_class = np.zeros(len(self.features_list))

        # Map back to feature names
        contributions = []
        for feat_name, shap_val in zip(self.features_list, shap_values_for_class):
            feat_val = X_sample[feat_name].iloc[0]
            contributions.append({
                "feature": feat_name,
                "val": float(feat_val),
                "contribution": float(shap_val)
            })
            
        # Sort by absolute contribution descending
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions

    def generate_global_summary_plot(self, X_background: pd.DataFrame) -> None:
        """Computes SHAP values on background/test samples and saves the global summary plot.

        Args:
            X_background: A DataFrame of samples (e.g. 200 samples) matching FeatureSet_B.
        """
        logger.info("Generating global SHAP summary plot...")
        X_background = X_background[self.features_list]
        
        shap_values = self.explainer.shap_values(X_background)
        
        # multi-class summary plot
        plt.figure(figsize=(10, 8))
        class_names = ["Low", "Medium", "High", "Critical"]
        
        shap.summary_plot(
            shap_values,
            X_background,
            class_names=class_names,
            show=False
        )
        plt.title("Global SHAP Feature Attribution (FeatureSet_B)")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "global_shap_summary.png", dpi=150)
        plt.close()
        logger.info("Global SHAP summary plot saved to %s", self.plots_dir / "global_shap_summary.png")
