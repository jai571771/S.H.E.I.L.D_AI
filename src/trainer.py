import time
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier
from src.utils import get_config, setup_logger, PROJECT_ROOT

logger = setup_logger("pipeline.trainer")

class ModelTrainer:
    """Trains baseline machine learning models and outputs serialization/comparison results."""

    def __init__(self, data_dir: Optional[str] = None, models_dir: Optional[str] = None) -> None:
        """Initializes the ModelTrainer."""
        config = get_config()
        self.data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / config["paths"]["data_dir"] / "FeatureSet_B"
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / config["paths"]["models_dir"]

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.xgb_params = config["model"]["xgb"]

    def train_and_compare(self) -> pd.DataFrame:
        """Trains baseline models on FeatureSet_B and returns a performance comparison DataFrame."""
        logger.info("Loading training and validation sets for model benchmarking...")
        
        X_train = pd.read_csv(self.data_dir / "X_train.csv")
        y_train = pd.read_csv(self.data_dir / "y_train.csv").values.ravel()
        X_val = pd.read_csv(self.data_dir / "X_valid.csv")
        y_val = pd.read_csv(self.data_dir / "y_valid.csv").values.ravel()

        models = {
            "Logistic_Regression": LogisticRegression(max_iter=200, random_state=42),
            "Random_Forest": RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
            "Extra_Trees": ExtraTreesClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(
                n_estimators=self.xgb_params["n_estimators"],
                max_depth=self.xgb_params["max_depth"],
                learning_rate=self.xgb_params["learning_rate"],
                eval_metric=self.xgb_params["eval_metric"],
                random_state=self.xgb_params["random_state"],
                n_jobs=-1
            )
        }

        results = []

        for name, model in models.items():
            logger.info("Training model: %s...", name)
            t0 = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - t0

            # Predictions
            y_pred = model.predict(X_val)
            
            # Metrics
            acc = accuracy_score(y_val, y_pred)
            _, _, f1_macro, _ = precision_recall_fscore_support(y_val, y_pred, average="macro", zero_division=0)
            
            # Critical Recall (Recall of class 3 = Critical)
            crit_mask = (y_val == 3)
            crit_recall = float((y_pred[crit_mask] == 3).sum() / crit_mask.sum()) if crit_mask.sum() > 0 else 0.0

            # False Negative Rate (predicted < actual for actual > 0)
            hazard_mask = (y_val > 0)
            fn_count = ((y_pred < y_val) & hazard_mask).sum()
            fnr = float(fn_count / hazard_mask.sum()) if hazard_mask.sum() > 0 else 0.0

            results.append({
                "Model_Name": name,
                "Accuracy": float(acc),
                "F1_Macro": float(f1_macro),
                "Critical_Recall": float(crit_recall),
                "False_Negative_Rate": float(fnr),
                "Fit_Time_Sec": float(fit_time)
            })
            logger.info("%s -> Acc: %.4f, F1: %.4f, Crit Recall: %.4f, FNR: %.4f", name, acc, f1_macro, crit_recall, fnr)

        results_df = pd.DataFrame(results)
        
        # Save XGBoost as the certified classifier
        logger.info("Serializing and saving certified XGBoost classifier to risk_classifier.pkl...")
        xgb_model = models["XGBoost"]
        joblib.dump(xgb_model, self.models_dir / "risk_classifier.pkl")
        logger.info("Model saved successfully.")

        return results_df
