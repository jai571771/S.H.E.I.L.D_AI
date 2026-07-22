import logging
import json
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from src.utils import get_config, setup_logger, PROJECT_ROOT

logger = setup_logger("pipeline.preprocessing")

class FeatureSchemaMismatchError(KeyError):
    """Raised when the input feature schema does not match the fitted preprocessor schema."""
    def __init__(self, missing_features: List[str], unexpected_features: List[str], message: str) -> None:
        super().__init__(message)
        self.missing_features = missing_features
        self.unexpected_features = unexpected_features


class PreprocessingPipeline:
    """Manages cleaning, encoding, scaling, splitting, and saving pipeline artifacts."""

    def __init__(
        self,
        models_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        reports_dir: Optional[str] = None
    ) -> None:
        """Initializes the PreprocessingPipeline and creates output folders."""
        config = get_config()
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / config["paths"]["models_dir"]
        self.data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / config["paths"]["data_dir"]
        self.reports_dir = Path(reports_dir) if reports_dir else PROJECT_ROOT / config["paths"]["reports_dir"]

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.inject_training_noise = config["preprocessing"]["inject_training_noise"]
        self.noise_std = config["preprocessing"]["noise_std"]

        self.encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[Any] = None
        self.scaling_strategy: str = "RobustScaler"

        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
        self.bool_cols: List[str] = []

        logger.info("PreprocessingPipeline initialized.")

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verifies no constant columns, sparse features, or infinite values.

        Args:
            df: Input raw feature DataFrame.

        Returns:
            A cleaned DataFrame.
        """
        logger.info("Starting Data Cleaning checks...")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Impute missing values
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].index.tolist()
        
        for col in df.columns:
            dtype_str = str(df[col].dtype).lower()
            # Categorical detection
            if df[col].dtype == object or "str" in dtype_str or "string" in dtype_str or col in ["Shift_Type", "PPE_Compliance", "Permit_Type", "CCTV_Event", "Event_Phase", "Event_ID"]:
                df[col] = df[col].fillna("None").astype(str)
            elif df[col].dtype == bool or "bool" in dtype_str:
                df[col] = df[col].fillna(False)
            else:
                col_median = df[col].median()
                df[col] = df[col].fillna(col_median if not pd.isna(col_median) else 0.0)

        # Detect constant columns (excluding timestamp) only for multi-row datasets
        constant_cols = []
        if len(df) > 1:
            constant_cols = [col for col in df.columns if df[col].nunique() <= 1 and col != "Timestamp"]
            if constant_cols:
                logger.info("Dropping constant columns: %s", constant_cols)
                df = df.drop(columns=constant_cols)

        # Generate data_cleaning_report.md
        cleaning_lines = [
            "# Data Cleaning Report",
            "",
            "## Summary Statistics",
            f"- **Initial Column Count**: {len(df.columns) + len(constant_cols)}",
            f"- **Columns with Missing Values**: {len(cols_with_nulls)}",
            f"- **Constant Columns Dropped**: {len(constant_cols)}",
            "",
            "## Details of Missing Values & Imputation",
            "| Column | Null Count | Imputation Strategy |",
            "| :--- | :---: | :--- |"
        ]
        if not cols_with_nulls:
            cleaning_lines.append("| None | 0 | No missing values detected in continuous sensor variables. |")
        else:
            for col in cols_with_nulls:
                if col in df.columns:
                    imp = "Imputed with string 'None'" if df[col].dtype == object else "Imputed with column median"
                    cleaning_lines.append(f"| `{col}` | {null_counts[col]} | {imp} |")
                else:
                    cleaning_lines.append(f"| `{col}` | {null_counts[col]} | Constant column dropped |")

        with open(self.reports_dir / "data_cleaning_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(cleaning_lines))

        logger.info("Data cleaning complete. Saved data_cleaning_report.md.")
        return df

    def fit_transform_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Fits encoders/scalers and transforms features.

        Args:
            df: Cleaned feature DataFrame.

        Returns:
            A tuple of (transformed_df, encoded_feature_names_list).
        """
        logger.info("Detecting feature types for encoding and scaling...")
        feature_df = df.drop(columns=["Timestamp"], errors="ignore")

        self.cat_cols = []
        self.bool_cols = []
        self.num_cols = []

        for col in feature_df.columns:
            dtype_str = str(feature_df[col].dtype).lower()
            if feature_df[col].dtype == bool or "bool" in dtype_str:
                self.bool_cols.append(col)
            elif feature_df[col].dtype == object or "str" in dtype_str or "string" in dtype_str or col in ["Shift_Type", "PPE_Compliance", "Permit_Type", "CCTV_Event", "Event_Phase", "Event_ID", "BF_State_ID", "BF_State_Name", "CO_State_ID", "CO_State_Name"]:
                self.cat_cols.append(col)
            else:
                self.num_cols.append(col)

        logger.info(
            "Feature type summary: Categorical = %d, Boolean = %d, Numeric = %d",
            len(self.cat_cols), len(self.bool_cols), len(self.num_cols)
        )

        encoded_dfs = []
        
        # 1. One-Hot Encoding for Categoricals
        if self.cat_cols:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded_arr = self.encoder.fit_transform(feature_df[self.cat_cols])
            encoded_names = self.encoder.get_feature_names_out(self.cat_cols)
            encoded_df = pd.DataFrame(encoded_arr, columns=encoded_names, index=feature_df.index)
            encoded_dfs.append(encoded_df)

        # 2. Boolean features as 1/0 integers
        bool_df = feature_df[self.bool_cols].astype(int)
        encoded_dfs.append(bool_df)

        # 3. Numeric scaling strategy selection
        numeric_df = feature_df[self.num_cols]
        q25 = numeric_df.quantile(0.25)
        q75 = numeric_df.quantile(0.75)
        iqr = q75 - q25
        outliers = ((numeric_df < (q25 - 1.5 * iqr)) | (numeric_df > (q75 + 1.5 * iqr))).sum().sum()
        outlier_ratio = float(outliers) / numeric_df.size if numeric_df.size > 0 else 0.0

        if outlier_ratio > 0.01:
            self.scaling_strategy = "RobustScaler"
            self.scaler = RobustScaler()
            logger.info("Outlier ratio is %.2f%% (>1%%). Selecting RobustScaler.", outlier_ratio * 100)
        else:
            self.scaling_strategy = "StandardScaler"
            self.scaler = StandardScaler()
            logger.info("Outlier ratio is %.2f%% (<=1%%). Selecting StandardScaler.", outlier_ratio * 100)

        scaled_arr = self.scaler.fit_transform(numeric_df)
        scaled_df = pd.DataFrame(scaled_arr, columns=self.num_cols, index=feature_df.index)
        encoded_dfs.append(scaled_df)

        transformed_df = pd.concat(encoded_dfs, axis=1)
        logger.info("Transformed DataFrame shape: %s", transformed_df.shape)
        
        return transformed_df, transformed_df.columns.tolist()

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a new dataset using the already fitted preprocessors."""
        feature_df = df.drop(columns=["Timestamp"], errors="ignore")

        # Validate schema before scaling/encoding
        expected_cols = self.cat_cols + self.bool_cols + self.num_cols
        if expected_cols:
            missing = [col for col in expected_cols if col not in feature_df.columns]
            unexpected = [col for col in feature_df.columns if col not in expected_cols]
            if missing:
                msg = (
                    f"Feature schema mismatch: Missing {len(missing)} expected columns "
                    f"in the input DataFrame. Missing columns: {missing}. "
                    f"Unexpected columns: {unexpected}. "
                    f"Suggested fix: Ensure raw telemetry contains all required sensor variables "
                    f"or rerun the training pipeline to synchronize the preprocessor."
                )
                raise FeatureSchemaMismatchError(missing, unexpected, msg)

        encoded_dfs = []
        if self.cat_cols and self.encoder:
            encoded_arr = self.encoder.transform(feature_df[self.cat_cols])
            encoded_names = self.encoder.get_feature_names_out(self.cat_cols)
            encoded_df = pd.DataFrame(encoded_arr, columns=encoded_names, index=feature_df.index)
            encoded_dfs.append(encoded_df)

        bool_df = feature_df[self.bool_cols].astype(int)
        encoded_dfs.append(bool_df)

        if self.num_cols and self.scaler:
            scaled_arr = self.scaler.transform(feature_df[self.num_cols])
            scaled_df = pd.DataFrame(scaled_arr, columns=self.num_cols, index=feature_df.index)
            encoded_dfs.append(scaled_df)

        return pd.concat(encoded_dfs, axis=1)

    def split_and_export(
        self,
        df: pd.DataFrame,
        target_series: pd.Series,
        feature_sets: Dict[str, List[str]]
    ) -> None:
        """Splits data into Stratified Train/Val/Test (70/15/15) and exports partitions."""
        class_mapping = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        y = target_series.map(class_mapping)

        # Stratified split: Train (70%) and Temp (30%)
        X_train_full, X_temp, y_train, y_temp = train_test_split(
            df, y, test_size=0.30, random_state=42, stratify=y
        )
        
        # Split temp into validation (15%) and test (15%)
        X_val_full, X_test_full, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        logger.info(
            "Stratified train/val/test split completed. Train: %d, Val: %d, Test: %d",
            len(y_train), len(y_val), len(y_test)
        )

        X_train_full = X_train_full.copy()
        X_val_full = X_val_full.copy()

        # Inject noise into continuous features for training/validation sets
        if self.inject_training_noise and self.noise_std > 0.0:
            logger.info("Injecting Gaussian noise (std=%.3f) into continuous features...", self.noise_std)
            rng = np.random.default_rng(seed=42)
            for col in self.num_cols:
                if col in X_train_full.columns:
                    X_train_full[col] = X_train_full[col] + rng.normal(0.0, self.noise_std, size=len(X_train_full))
                if col in X_val_full.columns:
                    X_val_full[col] = X_val_full[col] + rng.normal(0.0, self.noise_std, size=len(X_val_full))

        # Export for each feature set
        for set_name, feature_list in feature_sets.items():
            out_folder = self.data_dir / set_name
            out_folder.mkdir(parents=True, exist_ok=True)

            valid_features = [f for f in feature_list if f in df.columns]

            X_train_sub = X_train_full[valid_features]
            X_val_sub = X_val_full[valid_features]
            X_test_sub = X_test_full[valid_features]

            X_train_sub.to_csv(out_folder / "X_train.csv", index=False)
            X_val_sub.to_csv(out_folder / "X_valid.csv", index=False)
            X_test_sub.to_csv(out_folder / "X_test.csv", index=False)

            y_train.to_csv(out_folder / "y_train.csv", index=False, header=True)
            y_val.to_csv(out_folder / "y_valid.csv", index=False, header=True)
            y_test.to_csv(out_folder / "y_test.csv", index=False, header=True)

            logger.info("Exported CSV partitions to: %s", out_folder)

    def save_artifacts(self, feature_sets: Dict[str, List[str]], transformed_cols: List[str]) -> None:
        """Saves encoders, scalers, feature names, and configs to disk."""
        if self.encoder:
            joblib.dump(self.encoder, self.models_dir / "encoder.pkl")
        if self.scaler:
            joblib.dump(self.scaler, self.models_dir / "scaler.pkl")

        with open(self.models_dir / "selected_features.json", "w", encoding="utf-8") as f:
            json.dump(feature_sets, f, indent=4)

        with open(self.models_dir / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(transformed_cols, f, indent=4)

        config_meta = {
            "scaling_strategy": self.scaling_strategy,
            "categorical_columns": self.cat_cols,
            "boolean_columns": self.bool_cols,
            "numeric_columns": self.num_cols,
            "split_ratio": {"train": 0.70, "valid": 0.15, "test": 0.15}
        }
        with open(self.models_dir / "pipeline_config.json", "w", encoding="utf-8") as f:
            json.dump(config_meta, f, indent=4)

        logger.info("All preprocessing artifacts saved successfully to: %s", self.models_dir)
