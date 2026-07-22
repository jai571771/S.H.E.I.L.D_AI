# Root Cause Analysis (RCA) - Schema Consistency & Preprocessing Mismatch

## 1. Executive Summary
Following the Phase 2.6.1 leakage cleanup, a mismatch was observed between the active dataset schema, the serialized preprocessing artifacts, and the test/inference pipelines. This document details the investigation, root causes, and resolutions that restored complete architectural consistency across the Industrial Safety Intelligence Platform.

---

## 2. Failed Components & Root Causes

### Incident A: Spearman Correlation String Error (Pipeline Crash)
*   **Symptom**: Running `run_pipeline.py` threw `ValueError: could not convert string to float: 'Medium'` during feature ranking.
*   **Root Cause**: The raw target series `Future_Risk_Level` contained categorical string values (`Low`, `Medium`, `High`, `Critical`). The pandas `.corrwith(..., method='spearman')` method attempted to convert this string column to floats, causing an unhandled conversion error.
*   **Consequence**: The retraining pipeline crashed halfway and could never serialize updated preprocessors (`scaler.pkl`, `encoder.pkl`, etc.) matching the leakage-cleaned schema. This left the system with stale artifacts.

### Incident B: Artificially Constant Column Removal (Test Failure)
*   **Symptom**: `test_clean_and_transform` threw `KeyError: 'CO_Gas_Pressure'`.
*   **Root Cause**: The mock dataset in `test_preprocessing.py` contained only 2 rows, where `CO_Gas_Pressure` and `BF_Top_Temperature` were partially missing. Imputation with the median caused these columns to resolve to the same value across both rows, making them constant. The `clean_dataset()` function dropped them as "constant columns," but the test expected them to remain, causing a `KeyError`.

### Incident C: Telemetry Schema Mismatch (Inference Failure)
*   **Symptom**: `test_predict_single` crashed with a `KeyError` (later replaced by the defensive `FeatureSchemaMismatchError` showing 141 missing columns).
*   **Root Cause**: The unit test passed a hand-crafted `mock_telemetry` dictionary with only 15 continuous features. However, the fitted `PreprocessingPipeline` expected the full set of 163 numeric, categorical, and boolean columns. The lack of standard alignment between the test fixture and the dataset schema caused a crash during `transform_features()`.

---

## 3. Corrective Measures & Architectural Repairs

1.  **Robust Target Mapping & Feature Selection**:
    *   Modified `rank_features()` in `src/feature_selector.py` to detect non-numeric targets using `pd.api.types.is_numeric_dtype` and map them to `0, 1, 2, 3`.
    *   Purged any leftover non-numeric columns from the ranking DataFrame to guarantee zero string representation in correlation matrices.
2.  **Telemetry-Aware Column Dropping**:
    *   Restricted constant column dropping in `src/preprocessing.py` to cases where `len(df) > 1`. This prevents dropping valid columns during single-row inference.
3.  **Expanded Test Fixture**:
    *   Updated `tests/test_preprocessing.py` to use a 3-row mock dataset with distinct values to prevent columns from becoming artificially constant.
4.  **Dynamic Telemetry Test Input**:
    *   Updated `tests/test_inference.py` to load a real row from the raw dataset via `DatasetLoader`, ensuring the mock telemetry matches the exact schema expected by the preprocessor.
5.  **Defensive Validation**:
    *   Introduced `FeatureSchemaMismatchError` at the start of `transform_features()` to raise explicit, informative error messages before mathematical scaling.

---

## 4. Prevention Checklist
- [x] Run full pipeline test suite before and after any database schema migrations.
- [x] Use dynamic, loader-generated fixtures instead of hardcoded dictionaries for end-to-end integration tests.
- [x] Maintain identical target mapping rules between trainer, evaluator, and inference modules.
