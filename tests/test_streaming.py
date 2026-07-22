import unittest
import math
from datetime import datetime, timedelta
from src.streaming.telemetry_validator import TelemetryValidator

class TestTelemetryValidator(unittest.TestCase):
    """Unit tests for TelemetryValidator."""

    def setUp(self) -> None:
        self.validator = TelemetryValidator()
        self.base_telemetry = {
            "Timestamp": "2026-07-17 12:00:00",
            "BF_Blast_Flow": 1500.0,
            "BF_Top_Temperature": 120.0,
            "CO_Gas_Pressure": 2.2,
            "CO_CO": 1.5,
            "BF_CO": 14.5,
            "BF_Gas_Pressure": 1.8,
            "BF_H2": 1.1,
            "CO_NH3": 0.05,
            "BF_Blower_Current": 2800.0,
            "BF_Blower_Vibration": 1.2,
            "BF_Burden_Level": 2.1,
            "BF_Cooling_Water_Flow": 160.0,
            "BF_Cooling_Water_Temperature": 38.0,
            "BF_Top_Pressure": 1.4,
            "CO_Oven_Temperature": 1050.0,
            "CO_Flue_Temperature": 1180.0,
            "CO_Pusher_Current": 140.0,
            "CO_Pusher_Vibration": 1.1,
            "CO_Quenching_Water_Flow": 90.0,
            "CO_Steam_Temperature": 260.0,
            "Ambient_Temp": 27.0,
            "Worker_Count": 4,
            "Shift_Type": "Morning",
            "PPE_Compliance": "Yes",
            "Permit_Type": "None",
            "CCTV_Event": "Normal",
            "Maintenance_Active": False,
            "Gas_Test_Completed": True
        }

    def test_validator_valid_sample(self) -> None:
        is_valid, reason = self.validator.validate(self.base_telemetry)
        self.assertTrue(is_valid, f"Validation failed: {reason}")
        self.assertIsNone(reason)

    def test_validator_missing_timestamp(self) -> None:
        sample = self.base_telemetry.copy()
        sample.pop("Timestamp")
        is_valid, reason = self.validator.validate(sample)
        self.assertFalse(is_valid)
        self.assertIn("Missing 'Timestamp'", reason)

    def test_validator_duplicate_timestamp(self) -> None:
        # First call normalizes, setting the last timestamp state
        self.validator.normalize(self.base_telemetry)
        # Second call with identical or older timestamp should fail
        is_valid, reason = self.validator.validate(self.base_telemetry)
        self.assertFalse(is_valid)
        self.assertIn("Out-of-order or duplicate timestamp", reason)

    def test_validator_missing_fields(self) -> None:
        sample = self.base_telemetry.copy()
        sample.pop("BF_Blast_Flow")
        is_valid, reason = self.validator.validate(sample)
        self.assertFalse(is_valid)
        self.assertIn("Missing required telemetry fields", reason)

    def test_validator_normalization(self) -> None:
        sample = self.base_telemetry.copy()
        sample["PPE_Compliance"] = "No"
        sample["BF_Blast_Flow"] = "1750.5"  # Should parse string to float
        sample["CO_CO"] = None              # Should fallback to default
        
        normalized = self.validator.normalize(sample)
        self.assertFalse(normalized["PPE_Compliance"])
        self.assertEqual(normalized["BF_Blast_Flow"], 1750.5)
        self.assertEqual(normalized["CO_CO"], 1.5)  # default for CO_CO
        self.assertEqual(normalized["Timestamp"], "2026-07-17 12:00:00")

class TestTelemetryBuffer(unittest.TestCase):
    """Unit tests for TelemetryBuffer."""

    def setUp(self) -> None:
        from src.streaming.buffer import TelemetryBuffer
        self.buffer = TelemetryBuffer(max_size=5)

    def test_buffer_append_and_size(self) -> None:
        self.assertEqual(self.buffer.current_size(), 0)
        self.buffer.append({"val": 1})
        self.buffer.append({"val": 2})
        self.assertEqual(self.buffer.current_size(), 2)
        self.assertEqual(self.buffer.latest(), {"val": 2})

    def test_buffer_sliding_window_limit(self) -> None:
        for i in range(10):
            self.buffer.append({"val": i})
        self.assertEqual(self.buffer.current_size(), 5)
        # Oldest should have dropped; window should have last 5 elements [5, 6, 7, 8, 9]
        window = self.buffer.get_window(5)
        self.assertEqual(len(window), 5)
        self.assertEqual(window[0], {"val": 5})
        self.assertEqual(window[-1], {"val": 9})

    def test_buffer_dynamic_window(self) -> None:
        self.buffer.append({"val": 10})
        self.buffer.append({"val": 11})
        
        # We request 5 but only have 2. Under dynamic window startup, it returns 2.
        window = self.buffer.get_window(5)
        self.assertEqual(len(window), 2)
        self.assertEqual(window[0], {"val": 10})
        self.assertEqual(window[-1], {"val": 11})

    def test_buffer_is_ready(self) -> None:
        self.assertFalse(self.buffer.is_ready(3))
        self.buffer.append({"val": 1})
        self.buffer.append({"val": 2})
        self.assertFalse(self.buffer.is_ready(3))
        self.buffer.append({"val": 3})
        self.assertTrue(self.buffer.is_ready(3))

    def test_buffer_oldest_and_get_all(self) -> None:
        self.buffer.append({"val": 10})
        self.buffer.append({"val": 20})
        self.buffer.append({"val": 30})
        self.assertEqual(self.buffer.oldest(), {"val": 10})
        
        all_samples = self.buffer.get_all()
        self.assertEqual(len(all_samples), 3)
        self.assertEqual(all_samples[0], {"val": 10})
        self.assertEqual(all_samples[-1], {"val": 30})

    def test_buffer_clear(self) -> None:
        self.buffer.append({"val": 1})
        self.buffer.clear()
        self.assertEqual(self.buffer.current_size(), 0)
        self.assertIsNone(self.buffer.latest())
        self.assertIsNone(self.buffer.oldest())

class TestRollingStatisticsEngine(unittest.TestCase):
    """Unit tests for RollingStatisticsEngine."""

    def setUp(self) -> None:
        from src.streaming.buffer import TelemetryBuffer
        from src.streaming.rolling_stats import RollingStatisticsEngine
        self.buffer = TelemetryBuffer(max_size=30)
        self.engine = RollingStatisticsEngine(windows=[5, 15, 30])

    def test_rolling_stats_calculations(self) -> None:
        # Append samples with constant and increasing values
        for i in range(1, 11):
            sample = {s: float(i) for s in self.engine.physical_sensors}
            self.buffer.append(sample)
            stats = self.engine.update(self.buffer)
            
            # Check for key sensor BF_Blast_Flow
            if i >= 5:
                # Last 5 elements should be [i-4, i-3, i-2, i-1, i]
                # Mean should be i - 2
                self.assertAlmostEqual(stats["BF_Blast_Flow_roll_avg_5"], float(i - 2))
                # Gradient should be (current - start) / 5 = (i - (i-4)) / 5 = 4 / 5 = 0.8
                self.assertAlmostEqual(stats["BF_Blast_Flow_roll_grad_5"], 0.8)
                # Standard deviation of [1, 2, 3, 4, 5] is sqrt(2.0) = 1.414...
                if i == 5:
                    self.assertAlmostEqual(stats["BF_Blast_Flow_roll_std_5"], math.sqrt(2.0))

class TestDerivedFeatureEngine(unittest.TestCase):
    """Unit tests for DerivedFeatureEngine."""

    def setUp(self) -> None:
        from src.streaming.derived_features import DerivedFeatureEngine
        self.engine = DerivedFeatureEngine()
        self.telemetry = {
            "BF_Blast_Flow": 1500.0,
            "BF_Top_Temperature": 120.0,
            "CO_Gas_Pressure": 2.2,
            "CO_CO": 1.5,
            "BF_CO": 14.5,
            "BF_H2": 1.1,
            "CO_NH3": 0.05,
            "BF_Blower_Current": 2800.0,
            "BF_Blower_Vibration": 1.2,
            "BF_Burden_Level": 2.1,
            "BF_Cooling_Water_Flow": 160.0,
            "BF_Cooling_Water_Temperature": 38.0,
            "BF_Top_Pressure": 1.4,
            "CO_Oven_Temperature": 1050.0,
            "CO_Flue_Temperature": 1180.0,
            "CO_Pusher_Current": 140.0,
            "CO_Pusher_Vibration": 1.1,
            "CO_Quenching_Water_Flow": 90.0,
            "CO_Steam_Temperature": 260.0,
            "Ambient_Temp": 27.0,
            "Worker_Count": 4,
            "Shift_Type": "Morning",
            "PPE_Compliance": True,
            "Permit_Type": "None",
            "CCTV_Event": "Normal",
            "Maintenance_Active": False,
            "Gas_Test_Completed": True
        }
        self.rolling_stats = {
            "BF_Top_Pressure_roll_grad_5": 0.1,
            "BF_Blast_Flow_roll_grad_5": 10.0,
            "CO_Gas_Pressure_roll_grad_5": 0.2,
            "CO_Oven_Temperature_roll_grad_5": 5.0
        }

    def test_derived_indices_calculation(self) -> None:
        derived = self.engine.compute(self.telemetry, self.rolling_stats)
        
        # Check stability calculations
        self.assertTrue(0.0 <= derived["Process_Stability_Index"] <= 100.0)
        
        # Check gas risk (BF_CO: 14.5 is far below 40.0, BF_H2: 1.1 is below 3.0, so should be 0)
        self.assertEqual(derived["Gas_Risk_Index"], 0.0)

    def test_maintenance_risk(self) -> None:
        # Maintenance active + General permit
        telemetry = self.telemetry.copy()
        telemetry["Maintenance_Active"] = True
        telemetry["Permit_Type"] = "General"
        telemetry["Gas_Test_Completed"] = True
        telemetry["Shift_Type"] = "Morning"
        
        derived = self.engine.compute(telemetry, self.rolling_stats)
        # maintenance active default is 30.0 + permit general (0) + gas complete (0) + morning (0) = 30.0
        self.assertEqual(derived["Maintenance_Risk_Index"], 30.0)

    def test_feature_parity_with_dataset(self) -> None:
        """Verifies that derived indexes match historical dataset values within 1e-5 tolerance."""
        import pandas as pd
        from src.streaming.telemetry_validator import TelemetryValidator
        from src.streaming.buffer import TelemetryBuffer
        from src.streaming.rolling_stats import RollingStatisticsEngine
        from src.streaming.derived_features import DerivedFeatureEngine
        
        # Load dataset, parse and sort by Timestamp to restore chronological order,
        # then select first 50 rows for validation.
        df_all = pd.read_csv("data/industrial_safety_dataset_3.0.csv", keep_default_na=False)
        df_all["ts_dt"] = pd.to_datetime(df_all["Timestamp"])
        df = df_all.sort_values("ts_dt").head(50)
        
        validator = TelemetryValidator()
        buffer = TelemetryBuffer(max_size=30)
        stats_engine = RollingStatisticsEngine(windows=[5, 15, 30])
        derived_engine = DerivedFeatureEngine()
        
        for idx, row in df.iterrows():
            raw_sample = row.to_dict()
            
            # Extract only columns that are part of raw telemetry
            raw_keys = validator.RAW_SENSORS.union(validator.CATEGORICAL_FIELDS.keys()).union(validator.BOOLEAN_FIELDS)
            raw_telemetry = {k: raw_sample[k] for k in raw_keys if k in raw_sample}
            raw_telemetry["Timestamp"] = raw_sample["Timestamp"]
            
            # Run streaming pipeline step
            is_valid, reason = validator.validate(raw_telemetry)
            self.assertTrue(is_valid, f"Validation failed at row {idx}: {reason}")
            normalized = validator.normalize(raw_telemetry)
            
            # Feed Safety_Index from the historical dataset to mimic the rule engine's pre-calculation
            normalized["Safety_Index"] = raw_sample["Safety_Index"]
            
            buffer.append(normalized)
            rolling_stats = stats_engine.update(buffer)
            derived_features = derived_engine.compute(normalized, rolling_stats)
            
            # Only assert parity after 30-step warmup to let rolling windows fully populate
            if idx >= 30:
                indices_to_check = [
                    "Process_Stability_Index",
                    "Gas_Risk_Index",
                    "Equipment_Health_Index",
                    "Worker_Exposure_Index",
                    "Thermal_Risk_Index",
                    "Maintenance_Risk_Index",
                    "Safety_Index",
                    "Compound_Risk_Score",
                    "Data_Quality_Score"
                ]
                
                for index_name in indices_to_check:
                    expected_val = float(raw_sample[index_name])
                    actual_val = derived_features[index_name]
                    
                    # Check within tight tolerance of 1e-4
                    self.assertAlmostEqual(actual_val, expected_val, places=4, 
                                           msg=f"Parity mismatch on row {idx} for {index_name}: expected {expected_val}, got {actual_val}")

class TestOnlineFeatureStore(unittest.TestCase):
    """Unit tests for OnlineFeatureStore."""

    def setUp(self) -> None:
        from src.streaming.online_store import OnlineFeatureStore
        self.store = OnlineFeatureStore()

    def test_feature_alignment_and_encoding(self) -> None:
        # Prepare inputs
        telemetry = {
            "BF_Blast_Flow": 1500.0,
            "Worker_Count": 4,
            "Permit_Type": "Hot Work",
            "Maintenance_Active": True
        }
        rolling_stats = {
            "CO_CO_roll_avg_15": 2.2,
            "CO_CO_roll_avg_30": 2.5
        }
        derived_features = {
            "Worker_Exposure_Index": 35.0,
            "Gas_Risk_Index": 10.0
        }
        
        vector = self.store.serve_feature_vector(telemetry, rolling_stats, derived_features)
        
        # 1. Output keys must exactly match FeatureSet_B
        self.assertEqual(list(vector.keys()), self.store.selected_features)
        
        # 2. Check categorical one-hot encoding logic
        # Permit_Type was "Hot Work", so Permit_Type_Hot Work should be 1.0, Permit_Type_None should be 0.0
        self.assertEqual(vector.get("Permit_Type_Hot Work"), 1.0)
        self.assertEqual(vector.get("Permit_Type_None"), 0.0)
        
        # 3. Check boolean conversion
        # Maintenance_Active was True, so it should be float 1.0
        self.assertEqual(vector.get("Maintenance_Active"), 1.0)
        
        # 4. Check direct values
        self.assertEqual(vector.get("CO_CO_roll_avg_15"), 2.2)
        self.assertEqual(vector.get("Worker_Exposure_Index"), 35.0)

    def test_validation_errors(self) -> None:
        from src.streaming.online_store import FeatureSchemaMismatchError
        
        telemetry = {"Permit_Type": "Hot Work", "Maintenance_Active": True}
        rolling_stats = {}
        # Inject float('nan') into a key that is part of FeatureSet_B
        derived_features = {"Worker_Exposure_Index": float('nan')}
        
        with self.assertRaises(FeatureSchemaMismatchError):
            self.store.serve_feature_vector(telemetry, rolling_stats, derived_features)

    def test_get_schema(self) -> None:
        schema = self.store.get_schema()
        self.assertEqual(schema, self.store.selected_features)

class TestStreamHealthMonitor(unittest.TestCase):
    """Unit tests for StreamHealthMonitor."""

    def setUp(self) -> None:
        from src.streaming.stream_health import StreamHealthMonitor
        self.monitor = StreamHealthMonitor(latency_critical_threshold_ms=200.0, latency_warning_threshold_ms=50.0)

    def test_latency_health_transitions(self) -> None:
        # Initial status should be healthy (latencies initialized to 0.0)
        self.assertEqual(self.monitor.check_health(), "HEALTHY")
        
        # Under warning threshold (e.g. 30ms total)
        self.monitor.record_latency("total_end_to_end", 0.030)
        self.assertEqual(self.monitor.check_health(), "HEALTHY")
        
        # Trigger Warning (e.g. 60ms total)
        self.monitor.record_latency("total_end_to_end", 0.060)
        self.assertEqual(self.monitor.check_health(), "WARNING")
        
        # Trigger Critical (e.g. 210ms total)
        self.monitor.record_latency("total_end_to_end", 0.210)
        self.assertEqual(self.monitor.check_health(), "CRITICAL")

    def test_packet_loss_and_utilization(self) -> None:
        # Simulate 10 packets received, 1 failed validation
        for _ in range(9):
            self.monitor.record_packet_received()
        self.monitor.record_packet_received()
        self.monitor.record_validation_failure()
        
        self.monitor.record_buffer_utilization(15, 30)
        
        metrics = self.monitor.get_metrics()
        self.assertEqual(metrics["packets_received"], 10)
        self.assertEqual(metrics["packets_failed_validation"], 1)
        self.assertEqual(metrics["packet_loss_rate_pct"], 10.0)
        self.assertEqual(metrics["buffer_utilization_pct"], 50.0)

class TestEndToEndStreamingIntegration(unittest.TestCase):
    """Integration test verifying end-to-end execution of the streaming pipeline."""

    def test_pipeline_integration_flow(self) -> None:
        from src.streaming.telemetry_validator import TelemetryValidator
        from src.streaming.buffer import TelemetryBuffer
        from src.streaming.rolling_stats import RollingStatisticsEngine
        from src.streaming.derived_features import DerivedFeatureEngine
        from src.streaming.online_store import OnlineFeatureStore
        from src.streaming.stream_health import StreamHealthMonitor
        
        # Instantiate all components
        validator = TelemetryValidator()
        buffer = TelemetryBuffer(max_size=5)
        stats_engine = RollingStatisticsEngine(windows=[5])
        derived_engine = DerivedFeatureEngine()
        store = OnlineFeatureStore()
        monitor = StreamHealthMonitor()
        
        # Valid test sample base
        sample = {
            "Timestamp": "2026-07-17 12:00:00",
            "BF_Blast_Flow": 1500.0,
            "BF_Top_Temperature": 120.0,
            "CO_Gas_Pressure": 2.2,
            "CO_CO": 1.5,
            "BF_CO": 14.5,
            "BF_Gas_Pressure": 1.8,
            "BF_H2": 1.1,
            "CO_NH3": 0.05,
            "BF_Blower_Current": 2800.0,
            "BF_Blower_Vibration": 1.2,
            "BF_Burden_Level": 2.1,
            "BF_Cooling_Water_Flow": 160.0,
            "BF_Cooling_Water_Temperature": 38.0,
            "BF_Top_Pressure": 1.4,
            "CO_Oven_Temperature": 1050.0,
            "CO_Flue_Temperature": 1180.0,
            "CO_Pusher_Current": 140.0,
            "CO_Pusher_Vibration": 1.1,
            "CO_Quenching_Water_Flow": 90.0,
            "CO_Steam_Temperature": 260.0,
            "Ambient_Temp": 27.0,
            "Worker_Count": 4,
            "Shift_Type": "Morning",
            "PPE_Compliance": "Yes",
            "Permit_Type": "None",
            "CCTV_Event": "Normal",
            "Maintenance_Active": False,
            "Gas_Test_Completed": True
        }
        
        # Record and validate
        monitor.record_packet_received(sample["Timestamp"])
        is_valid, reason = validator.validate(sample)
        self.assertTrue(is_valid, f"Validation failed: {reason}")
        
        normalized = validator.normalize(sample)
        # Inject rule engine fallback Safety_Index
        normalized["Safety_Index"] = 100.0
        
        buffer.append(normalized)
        monitor.record_buffer_utilization(buffer.current_size(), buffer._max_size)
        
        # Compute stats
        rolling_stats = stats_engine.update(buffer)
        
        # Compute derived
        derived = derived_engine.compute(normalized, rolling_stats)
        
        # Assemble feature vector
        vector = store.serve_feature_vector(normalized, rolling_stats, derived)
        
        # Verify schema
        self.assertEqual(len(vector), len(store.selected_features))
        self.assertEqual(list(vector.keys()), store.selected_features)
        
        # Verify health metrics
        metrics = monitor.get_metrics()
        self.assertEqual(metrics["packets_received"], 1)
        self.assertEqual(metrics["packets_failed_validation"], 0)
        self.assertEqual(metrics["status"], "HEALTHY")


class TestRecommendationNormalization(unittest.TestCase):
    """Unit tests for recommendation normalization in the telemetry replay runner."""

    def test_legacy_recommendation_format(self) -> None:
        from telemetry_replay import normalize_recommendation
        legacy_rec = {
            "priority": "Critical",
            "action": "Shutdown the coke oven pusher motor.",
            "reason": "Vibration levels exceeded 18mm/s",
            "category": "Coke Oven"
        }
        normalized = normalize_recommendation(legacy_rec)
        self.assertEqual(normalized["priority"], "Critical")
        self.assertEqual(normalized["action"], "Shutdown the coke oven pusher motor.")
        self.assertEqual(normalized["reason"], "Vibration levels exceeded 18mm/s")
        self.assertEqual(normalized["category"], "Coke Oven")

    def test_new_recommendation_format(self) -> None:
        from telemetry_replay import normalize_recommendation
        new_rec = {
            "rule_id": "R004",
            "rule_name": "Worker Gas Exposure",
            "process_area": "Blast Furnace",
            "hazard_category": "Toxic Exposure",
            "description": "Worker directly exposed to increasing CO concentration",
            "severity": "High",
            "recommended_action": "Evacuate workers and use breathing apparatus"
        }
        normalized = normalize_recommendation(new_rec)
        self.assertEqual(normalized["priority"], "High")
        self.assertEqual(normalized["action"], "Evacuate workers and use breathing apparatus")
        self.assertEqual(normalized["reason"], "Worker directly exposed to increasing CO concentration")
        self.assertEqual(normalized["category"], "Toxic Exposure")

    def test_missing_priority(self) -> None:
        from telemetry_replay import normalize_recommendation
        # Should fallback to 'severity', or default to 'HIGH' if both missing
        rec = {
            "action": "Verify compliance status.",
            "description": "Worker count is non-zero without active supervisor.",
            "hazard_category": "Compliance"
        }
        normalized = normalize_recommendation(rec)
        self.assertEqual(normalized["priority"], "HIGH")
        self.assertEqual(normalized["action"], "Verify compliance status.")

    def test_missing_action(self) -> None:
        from telemetry_replay import normalize_recommendation
        rec = {
            "priority": "Low",
            "description": "Temperature slightly elevated"
        }
        with self.assertRaises(ValueError):
            normalize_recommendation(rec)

    def test_malformed_recommendation_objects(self) -> None:
        from telemetry_replay import normalize_recommendation
        with self.assertRaises(ValueError):
            normalize_recommendation("Not a dictionary")
        with self.assertRaises(ValueError):
            normalize_recommendation([])
        with self.assertRaises(ValueError):
            normalize_recommendation(None)

    def test_empty_recommendation_list(self) -> None:
        # Replays with empty lists should not trigger any prints or normalization failures
        from telemetry_replay import print_dashboard
        pred_res = {"recommendations": []}
        derived = {}
        health = {
            "status": "HEALTHY",
            "packets_received": 0,
            "buffer_utilization_pct": 0.0,
            "buffer_size": 0,
            "buffer_capacity": 30,
            "packet_loss_rate_pct": 0.0,
            "critical_threshold_ms": 200.0,
            "latest_latencies_ms": {
                "validation": 0, "rolling_stats": 0, "derived_features": 0, "feature_store": 0, "inference": 0, "total_end_to_end": 0
            },
            "average_latencies_ms": {
                "validation": 0, "rolling_stats": 0, "derived_features": 0, "feature_store": 0, "inference": 0, "total_end_to_end": 0
            }
        }
        # Call it, expecting it not to raise any exception or crash
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            print_dashboard(0, {}, pred_res, derived, health)
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    unittest.main()
