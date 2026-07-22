# Unit Test Changes Report

## 1. Summary of Changes
To restore architectural consistency after Phase 2.6.1, we audited and repaired two failing unit tests without compromising or bypassing any safety assertions.

---

## 2. Detailed Test Audit & Rationale

### Test Case A: `test_preprocessing.py` (`test_clean_and_transform`)
*   **Initial Failure**: `KeyError: 'CO_Gas_Pressure'` during assertion check `self.assertFalse(cleaned_df["CO_Gas_Pressure"].isnull().any())`.
*   **Resolution Rationale**: 
    *   *Implementation vs. Test*: The preprocessing code was correct in detecting and dropping constant columns. However, the mock dataset in the unit test was too small (only 2 rows), which caused the variables `CO_Gas_Pressure` and `BF_Top_Temperature` to become constant (single unique non-null value) after median imputation.
    *   *Fix Applied*: Expanded the mock dataset in `tests/test_preprocessing.py` from 2 rows to 3 rows, introducing distinct numeric readings for all columns. This prevents them from becoming artificially constant after imputation, matching realistic dataset conditions where continuous sensor values are never perfectly constant. The length assertions were updated to expect 3 rows.

### Test Case B: `test_inference.py` (`test_predict_single`)
*   **Initial Failure**: `FeatureSchemaMismatchError` (specifically missing 141 expected columns).
*   **Resolution Rationale**:
    *   *Implementation vs. Test*: The preprocessor implementation was correct in enforcing that all 163 active dataset features are present before invoking the scaler. The test was incorrect because it fed a manual, sparse 15-key dictionary that did not represent a real telemetry row.
    *   *Fix Applied*: Replaced the hardcoded, incomplete dictionary in `tests/test_inference.py` with dynamic loading. The test now utilizes `DatasetLoader().load_and_validate()` to extract a single row from the raw dataset, pop off the target variable, and use that as the mock telemetry. This guarantees 100% schema alignment with the fitted preprocessing models and ensures the test reflects real-world server deployment where a full telemetry row is received.

---

## 3. Execution Verification
All tests are validated and pass successfully:
```powershell
python -m unittest discover -s tests -p "test_*.py"
Ran 5 tests in 6.964s
OK
```
