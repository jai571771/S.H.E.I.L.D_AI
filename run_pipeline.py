import sys
from pathlib import Path
import pandas as pd
from src.utils import get_config, setup_logger
from src.dataset_loader import DatasetLoader
from src.feature_selector import FeatureSelector
from src.preprocessing import PreprocessingPipeline
from src.trainer import ModelTrainer
from src.evaluation import ModelEvaluator
from src.explainability import ShapExplainerManager

logger = setup_logger("run_pipeline")

def main():
    logger.info("==============================================")
    logger.info("STARTING INDUSTRIAL SAFETY AI PIPELINE RETRAIN")
    logger.info("==============================================")

    # 1. Dataset Loading
    loader = DatasetLoader()
    df, meta = loader.load_and_validate()
    if df is None or not meta["success"]:
        logger.error("Dataset loading failed. Aborting pipeline.")
        sys.exit(1)

    # 2. Leakage Removal
    selector = FeatureSelector()
    df_clean, removed_cols = selector.remove_leakage(df)
    
    # 3. Categorize remaining features
    selector.categorize_features(df_clean)

    # 4. Fit Preprocessing Pipeline on full dataset to get target mapped correctly
    target_series = df["Future_Risk_Level"]
    
    # Preprocessor initialization
    preprocessor = PreprocessingPipeline()
    cleaned_df = preprocessor.clean_dataset(df_clean)
    
    trans_df, transformed_cols = preprocessor.fit_transform_features(cleaned_df)

    # 5. Feature Ranking (using numeric-transformed features)
    ranked_features = selector.rank_features(trans_df, target_series)
    
    # Generate feature sets list
    feature_sets = selector.create_feature_sets(ranked_features)

    # 6. Preprocessing Train/Val/Test Split and Noise Injection
    preprocessor.split_and_export(trans_df, target_series, feature_sets)
    preprocessor.save_artifacts(feature_sets, transformed_cols)

    # 7. Model Training and Benchmarking
    trainer = ModelTrainer()
    benchmark_df = trainer.train_and_compare()
    logger.info("Model benchmarking results:\n%s", benchmark_df.to_string(index=False))

    # 8. Test Set Evaluation
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()

    # 9. Global SHAP Summary Plot
    # Load background data from test set for SHAP summary
    logger.info("Generating global SHAP summary plot on background test set...")
    try:
        X_test = pd.read_csv(Path(preprocessor.data_dir) / "FeatureSet_B" / "X_test.csv")
        shap_manager = ShapExplainerManager()
        # Use 100 samples from the test set as background for the global summary plot
        background_samples = X_test.sample(n=min(100, len(X_test)), random_state=42)
        shap_manager.generate_global_summary_plot(background_samples)
    except Exception as e:
        logger.exception("Global SHAP summary plot generation failed: %s", str(e))

    logger.info("================================================")
    logger.info("PIPELINE TRAINING AND CERTIFICATION COMPLETED")
    logger.info("================================================")

if __name__ == "__main__":
    main()
