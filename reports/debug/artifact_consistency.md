# Artifact Consistency Audit

## 1. Context & Scope
After Phase 2.6.1 leakage features were removed, old serialized artifacts in the `models/` directory were mismatched. This audit confirms that the artifacts have been regenerated and are aligned with the new leakage-free schema.

---

## 2. Serialized Preprocessing Artifacts List
All artifacts are located in the `models/preprocessing/` directory. Below is the verification status for each file:

| Artifact Name | Type | Size (Bytes) | Verification Status | Role in Pipeline |
| :--- | :--- | :--- | :---: | :--- |
| `scaler.pkl` | Binary (RobustScaler) | 7,447 | **Active & Fresh** | Fits/Scales 150 continuous features. |
| `encoder.pkl` | Binary (OneHotEncoder) | 1,717 | **Active & Fresh** | Encodes categorical features (`PPE_Compliance`, `Shift_Type`, etc.) |
| `risk_classifier.pkl` | Binary (XGBoost Model) | 724,168 | **Active & Fresh** | Multi-class risk prediction engine (4 levels). |
| `selected_features.json` | JSON Config | 10,667 | **Active & Fresh** | Catalog of FeatureSet A, B, and C lists. |
| `feature_names.json` | JSON List | 5,562 | **Active & Fresh** | Ordered list of expected transformed input column names. |
| `pipeline_config.json` | JSON Config | 6,196 | **Active & Fresh** | Metadata including scaler strategy, type mappings, and splits. |

---

## 3. Consistency Checks

*   **Pre-existing Artifacts Purged**: Checked. All new artifacts were generated at timestamp `2026-07-17 12:28:07` following the successful execution of `run_pipeline.py`.
*   **Scale and Dimension Alignment**: Checked.
    *   Continuous features scaled in `scaler.pkl`: **150**
    *   Categorical features encoded in `encoder.pkl`: **4** (`PPE_Compliance`, `Shift_Type`, `Permit_Type`, `CCTV_Event`)
    *   Total columns in `feature_names.json` matching input to `risk_classifier.pkl` (for FeatureSet_B): **80**
*   **Inference Alignment**: The `InferenceEngine` dynamically loads `scaler.pkl`, `encoder.pkl`, `feature_names.json`, and `risk_classifier.pkl` to run `predict_single()`, which now executes with 0 KeyErrors on the updated schema.
