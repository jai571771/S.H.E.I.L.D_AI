import json
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from src.utils import get_config, setup_logger, PROJECT_ROOT
from src.preprocessing import PreprocessingPipeline
from src.explainability import ShapExplainerManager
from src.recommendations import SafetyRecommendationEngine

logger = setup_logger("pipeline.inference")

class InferenceEngine:
    """Combines preprocessing, model execution, SHAP attribution, and rule recommendation for single samples."""

    def __init__(self, models_dir: Optional[str] = None) -> None:
        """Initializes the InferenceEngine and pre-loads all models/pipelines."""
        config = get_config()
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / config["paths"]["models_dir"]
        self.class_names = ["Low", "Medium", "High", "Critical"]

        # 1. Load pipeline configurations
        config_path = self.models_dir / "pipeline_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found at: {config_path}. Please run training first.")
        with open(config_path, "r", encoding="utf-8") as f:
            self.pipeline_config = json.load(f)

        # 2. Load feature set B names
        feat_path = self.models_dir / "selected_features.json"
        if not feat_path.exists():
            raise FileNotFoundError(f"Selected features metadata not found at: {feat_path}")
        with open(feat_path, "r", encoding="utf-8") as f:
            feat_dict = json.load(f)
        self.selected_features = feat_dict["FeatureSet_B"]

        # 3. Load preprocessors
        self.encoder = joblib.load(self.models_dir / "encoder.pkl") if (self.models_dir / "encoder.pkl").exists() else None
        self.scaler = joblib.load(self.models_dir / "scaler.pkl") if (self.models_dir / "scaler.pkl").exists() else None

        # Instantiate preprocessing pipeline in transform mode
        self.prep_pipeline = PreprocessingPipeline(models_dir=str(self.models_dir))
        self.prep_pipeline.encoder = self.encoder
        self.prep_pipeline.scaler = self.scaler
        self.prep_pipeline.cat_cols = self.pipeline_config["categorical_columns"]
        self.prep_pipeline.bool_cols = self.pipeline_config["boolean_columns"]
        self.prep_pipeline.num_cols = self.pipeline_config["numeric_columns"]

        # 4. Load XGBoost Classifier
        model_path = self.models_dir / "risk_classifier.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Risk classifier not found at: {model_path}")
        self.classifier = joblib.load(model_path)

        # 5. Load SHAP explainer
        self.shap_manager = ShapExplainerManager(models_dir=str(self.models_dir))
        
        # 6. Load Recommendation Engine
        self.reco_engine = SafetyRecommendationEngine()

        logger.info("InferenceEngine successfully initialized and all models loaded.")

    def predict_single(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Runs end-to-end real-time inference on a single SCADA sample.

        Args:
            telemetry: Raw telemetry dictionary with sensor readings and context flags.

        Returns:
            A dictionary containing prediction class, confidence, probabilities, SHAP values, and recommendations.
        """
        # Convert dictionary to a single-row DataFrame
        df_raw = pd.DataFrame([telemetry])

        # Clean the sample using pipeline rules (e.g. median/None imputation)
        df_clean = self.prep_pipeline.clean_dataset(df_raw)

        # Encode and Scale features
        df_trans = self.prep_pipeline.transform_features(df_clean)

        # Align columns to FeatureSet_B list (fill missing columns with 0 if any)
        for col in self.selected_features:
            if col not in df_trans.columns:
                df_trans[col] = 0.0
        X_infer = df_trans[self.selected_features]

        # Model Inference
        probs = self.classifier.predict_proba(X_infer)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        # Build probabilities breakdown
        prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}

        # Calculate SHAP local contributions for the predicted class
        try:
            shap_list = self.shap_manager.explain_sample(X_infer, pred_idx)
            # Filter and take top 5 features
            shap_top = shap_list[:5]
        except Exception as e:
            logger.error("SHAP local contribution calculation failed: %s", str(e))
            shap_top = []

        # Evaluate safety recommendations
        recommendations = self.reco_engine.evaluate(telemetry)

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "shap_contributions": shap_top,
            "recommendations": recommendations
        }
