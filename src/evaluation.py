import logging
import joblib
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from src.utils import get_config, setup_logger, PROJECT_ROOT

logger = setup_logger("pipeline.evaluation")

class ModelEvaluator:
    """Evaluates the certified model on the held-out test partition and outputs charts."""

    def __init__(self, data_dir: Optional[str] = None, models_dir: Optional[str] = None, reports_dir: Optional[str] = None) -> None:
        """Initializes the ModelEvaluator."""
        config = get_config()
        self.data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / config["paths"]["data_dir"] / "FeatureSet_B"
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / config["paths"]["models_dir"]
        self.reports_dir = Path(reports_dir) if reports_dir else PROJECT_ROOT / config["paths"]["reports_dir"]
        
        self.plots_dir = self.reports_dir / "plots" / "test_evaluation"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = ["Low", "Medium", "High", "Critical"]

    def run_evaluation(self) -> None:
        """Runs test evaluation, saves plots and writes the markdown report."""
        logger.info("Loading certified model and test partition...")
        
        model_path = self.models_dir / "risk_classifier.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Certified classifier not found at {model_path}. Please train the model first.")
            
        model = joblib.load(model_path)
        
        X_test = pd.read_csv(self.data_dir / "X_test.csv")
        y_test = pd.read_csv(self.data_dir / "y_test.csv").values.ravel()

        logger.info("Running predictions on test set...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        y_conf = np.max(y_prob, axis=1)

        # 1. Classification Report
        clf_rep_dict = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
        clf_rep_text = classification_report(y_test, y_pred, target_names=self.class_names)

        # 2. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("XGBoost Test Set Confusion Matrix")
        plt.ylabel("Actual Risk Level")
        plt.xlabel("Predicted Risk Level")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        # 3. ROC-AUC (OvR)
        roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr")

        # 4. Precision-Recall Curves
        plt.figure(figsize=(8, 6))
        for i, c_name in enumerate(self.class_names):
            y_test_bin = (y_test == i).astype(int)
            prob_c = y_prob[:, i]
            precision, recall, _ = precision_recall_curve(y_test_bin, prob_c)
            ap = average_precision_score(y_test_bin, prob_c)
            plt.plot(recall, precision, label=f"{c_name} (AP = {ap:.4f})")
        plt.title("Test Set Precision-Recall Curves")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "precision_recall_curves.png", dpi=150)
        plt.close()

        # 5. Calibration Curves
        plt.figure(figsize=(8, 6))
        for i, c_name in enumerate(self.class_names):
            y_test_bin = (y_test == i).astype(int)
            prob_c = y_prob[:, i]
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test_bin, prob_c, n_bins=10, strategy="uniform"
            )
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=c_name)
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.title("Test Set Calibration Curves (OvR)")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "calibration_curves.png", dpi=150)
        plt.close()

        # 6. Probability Distributions
        plt.figure(figsize=(10, 6))
        for i, c_name in enumerate(self.class_names):
            sns.kdeplot(y_prob[:, i], label=c_name, fill=True, alpha=0.15)
        plt.title("Test Set Predicted Probability Distributions")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "probability_distribution.png", dpi=150)
        plt.close()

        # 7. Confidence Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(y_conf, bins=20, color="teal", edgecolor="black", alpha=0.7)
        plt.title("Test Set Prediction Confidence Histogram")
        plt.xlabel("Confidence (Max Class Probability)")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "confidence_histogram.png", dpi=150)
        plt.close()

        # 8. Write Markdown Report
        report_lines = [
            "# Test Set Metrics Report",
            "",
            "This report summarizes the performance metrics of the trained XGBoost model evaluated exclusively on the **Test Set partition** of the optimal **FeatureSet_B**.",
            "",
            "## 1. Classification Performance Overview",
            f"- **Test ROC-AUC (One-vs-Rest)**: **{roc_auc_ovr:.6f}**",
            "",
            "### Test Classification Report",
            "```",
            clf_rep_text,
            "```",
            "",
            "### Per-Class Test Metrics Breakdown",
            "| Risk Class | Precision | Recall | F1-Score | Support |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ]
        for c_name in self.class_names:
            metrics = clf_rep_dict[c_name]
            report_lines.append(
                f"| **{c_name}** | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {int(metrics['support'])} |"
            )
            
        report_lines.extend([
            "",
            "## 2. Confusion Matrix",
            "| Actual \\ Predicted | Low | Medium | High | Critical |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| **Low** | {cm[0,0]} | {cm[0,1]} | {cm[0,2]} | {cm[0,3]} |",
            f"| **Medium** | {cm[1,0]} | {cm[1,1]} | {cm[1,2]} | {cm[1,3]} |",
            f"| **High** | {cm[2,0]} | {cm[2,1]} | {cm[2,2]} | {cm[2,3]} |",
            f"| **Critical** | {cm[3,0]} | {cm[3,1]} | {cm[3,2]} | {cm[3,3]} |",
            "",
            "![Test Confusion Matrix](plots/test_evaluation/confusion_matrix.png)",
            "",
            "## 3. Precision–Recall and Calibration Diagnostics",
            "Precision-Recall evaluates model quality under class imbalance. Calibration curves verify if predicted probabilities represent true frequencies.",
            "",
            "![Precision-Recall Curves](plots/test_evaluation/precision_recall_curves.png)",
            "![Calibration Curves](plots/test_evaluation/calibration_curves.png)",
            "",
            "## 4. Probability and Confidence Distributions",
            "Analyze prediction confidence separation and class probability profiles on unseen test data.",
            "",
            "![Probability Distributions](plots/test_evaluation/probability_distribution.png)",
            "![Confidence Histogram](plots/test_evaluation/confidence_histogram.png)",
            "",
            "---",
            "**Report generated successfully. No training metrics are included to prevent reporting bias.**"
        ])

        report_path = self.reports_dir / "test_metrics_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info("Evaluation complete! Report written to %s", report_path)
