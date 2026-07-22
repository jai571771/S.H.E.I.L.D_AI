import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from src.utils import get_config, setup_logger, PROJECT_ROOT

logger = setup_logger("pipeline.feature_selector")

class FeatureSelector:
    """Manages leakage removal, feature cataloging, multi-method ranking, and feature set generation."""

    def __init__(self, reports_dir: Optional[str] = None) -> None:
        """Initializes the FeatureSelector using configs."""
        config = get_config()
        if reports_dir is None:
            self.reports_dir = PROJECT_ROOT / config["paths"]["reports_dir"]
        else:
            self.reports_dir = Path(reports_dir)
            
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Define target leak and non-predictive high-cardinality metadata columns to exclude
        self.leakage_columns = [
            "Safety_Index", "Compound_Risk_Score", "Future_Risk_Level", "Event_ID",
            "Cumulative_Risk_5", "Cumulative_Risk_15", "Cumulative_Risk_30",
            "Minutes_To_Incident", "Event_Phase",
            "BF_State_ID", "BF_State_Name", "BF_State_Risk",
            "CO_State_ID", "CO_State_Name", "CO_State_Risk"
        ] + [f"Rule_R{i:03d}" for i in range(1, 21)]

        logger.info("FeatureSelector initialized. Leakage columns to remove: %d", len(self.leakage_columns))

    def remove_leakage(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Removes target leakage columns from the dataframe.

        Args:
            df: Raw input DataFrame.

        Returns:
            A tuple of (clean_df, removed_columns_list).
        """
        removed = [col for col in self.leakage_columns if col in df.columns]
        clean_df = df.drop(columns=removed, errors="ignore")
        
        # Write feature_removal_report.md
        report_lines = [
            "# Target Leakage Removal Report",
            "",
            "The following variables were removed from the AI training feature set because they introduce direct mathematical target leakage or represent non-predictive high-cardinality metadata:",
            "",
            "| Removed Column | Leakage Category | Engineering Explanation |",
            "| :--- | :--- | :--- |",
            "| `Future_Risk_Level` | Target | The classification target itself. |",
            "| `Safety_Index` | Direct Class Boundary | Derived in the Rule Engine and directly thresholds risk levels (Low/Med/High/Crit). |",
            "| `Compound_Risk_Score` | Derived Target | Synthesized max-weighted sensor risk deviation score representing safety health. |",
            "| `Event_ID` | High Cardinality Metadata | Unique event trace identifier string; has no generalizable predictive value. |"
        ]
        for col in [c for c in removed if c.startswith("Rule_")]:
            report_lines.append(f"| `{col}` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |")
            
        with open(self.reports_dir / "feature_removal_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info("Removed %d target leakage columns. Saved feature_removal_report.md.", len(removed))
        return clean_df, removed

    def categorize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorizes remaining features and writes feature_catalog.csv.

        Args:
            df: Leakage-free DataFrame.

        Returns:
            A DataFrame representing the feature catalog.
        """
        catalog = []
        for col in df.columns:
            if col == "Timestamp" or col == "Future_Risk_Level":
                continue

            # Identify domain category
            if "roll_avg" in col or "roll_grad" in col or "Cumulative_Risk" in col:
                domain = "Rolling Features"
            elif "Risk_Index" in col or "Stability_Index" in col or "Lead_Indicator" in col or "Window" in col or "Event_" in col:
                domain = "Engineered Features"
            elif "BF_" in col:
                domain = "Blast Furnace"
            elif "CO_" in col:
                domain = "Coke Oven"
            elif "Worker" in col or "PPE" in col:
                domain = "Worker"
            elif "Maintenance" in col or "Permit" in col or "Gas_Test" in col:
                domain = "Maintenance"
            elif "Ambient" in col:
                domain = "Environment"
            else:
                domain = "General Context"

            # Identify data type
            dtype_str = str(df[col].dtype)
            if df[col].dtype == bool:
                data_type = "Boolean"
            elif df[col].dtype in [object, "category"] or col == "Shift_Type" or col == "PPE_Compliance":
                data_type = "Categorical"
            elif "int" in dtype_str:
                data_type = "Integer"
            elif "float" in dtype_str:
                data_type = "Float"
            else:
                data_type = "Numeric"

            catalog.append({
                "Feature_Name": col,
                "Domain_Area": domain,
                "Data_Type": data_type,
                "Null_Count": int(df[col].isnull().sum()),
                "Unique_Values": int(df[col].nunique())
            })

        catalog_df = pd.DataFrame(catalog)
        catalog_df.to_csv(self.reports_dir / "feature_catalog.csv", index=False)
        logger.info("Feature catalog generated with %d features. Saved to feature_catalog.csv.", len(catalog_df))
        return catalog_df

    def rank_features(self, df: pd.DataFrame, target_series: pd.Series) -> pd.DataFrame:
        """Ranks all features using 5 independent methods.

        Args:
            df: Leakage-free, numeric-encoded feature DataFrame.
            target_series: Target labels Series.

        Returns:
            A DataFrame representing the ranked features.
        """
        logger.info("Ranking features using multi-criteria selector (Variance, Spearman, MI, Random Forest, Extra Trees)...")
        
        # Map target series to numeric if non-numeric
        if not pd.api.types.is_numeric_dtype(target_series):
            class_mapping = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
            target_series = target_series.map(class_mapping).fillna(0).astype(int)

        # Drop any remaining non-numeric columns from df to prevent correlation errors
        non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric_cols:
            logger.warning("Dropping non-numeric columns from ranking DataFrame: %s", non_numeric_cols)
            df = df.drop(columns=non_numeric_cols)

        # Sample 5,000 rows for mutual information and ensemble calculations to speed up execution
        sample_size = min(5000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        sample_target = target_series.loc[sample_df.index]

        # 1. Variance
        variances = df.var()

        # 2. Spearman Correlation with target
        correlations = df.corrwith(target_series, method="spearman").abs().fillna(0)

        # 3. Mutual Information (using sample for speed)
        mi_values = mutual_info_classif(sample_df, sample_target, random_state=42)
        mi_series = pd.Series(mi_values, index=df.columns)

        # 4. Random Forest Gini Importance (using sample)
        rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(sample_df, sample_target)
        rf_importance = pd.Series(rf.feature_importances_, index=df.columns)

        # 5. Extra Trees Importance (using sample)
        et = ExtraTreesClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        et.fit(sample_df, sample_target)
        et_importance = pd.Series(et.feature_importances_, index=df.columns)

        # Build Rank Matrix
        rank_df = pd.DataFrame(index=df.columns)
        
        rank_df["Variance"] = variances
        rank_df["Variance_Rank"] = variances.rank(ascending=False, method="min")
        
        rank_df["Correlation"] = correlations
        rank_df["Correlation_Rank"] = correlations.rank(ascending=False, method="min")

        rank_df["Mutual_Information"] = mi_series
        rank_df["MI_Rank"] = mi_series.rank(ascending=False, method="min")

        rank_df["RF_Importance"] = rf_importance
        rank_df["RF_Rank"] = rf_importance.rank(ascending=False, method="min")

        rank_df["ET_Importance"] = et_importance
        rank_df["ET_Rank"] = et_importance.rank(ascending=False, method="min")

        # Average Rank
        rank_cols = ["Variance_Rank", "Correlation_Rank", "MI_Rank", "RF_Rank", "ET_Rank"]
        rank_df["Average_Rank"] = rank_df[rank_cols].mean(axis=1)
        
        ranked_final = rank_df.sort_values(by="Average_Rank", ascending=True).reset_index()
        ranked_final.rename(columns={"index": "Feature_Name"}, inplace=True)
        
        ranked_final.to_csv(self.reports_dir / "feature_ranking.csv", index=False)
        logger.info("Feature ranking successfully saved to feature_ranking.csv.")
        return ranked_final

    def create_feature_sets(self, ranked_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Creates three feature list sets (A: Full, B: Medium, C: Compact)."""
        all_features = ranked_df["Feature_Name"].tolist()
        num_features = len(all_features)
        
        set_a = all_features.copy()
        target_b_size = min(80, num_features)
        set_b = all_features[:target_b_size]
        target_c_size = min(45, num_features)
        set_c = all_features[:target_c_size]

        logger.info(
            "Feature sets defined: Set A (Full) = %d, Set B (Medium) = %d, Set C (Compact) = %d features",
            len(set_a), len(set_b), len(set_c)
        )
        return {
            "FeatureSet_A": set_a,
            "FeatureSet_B": set_b,
            "FeatureSet_C": set_c
        }
