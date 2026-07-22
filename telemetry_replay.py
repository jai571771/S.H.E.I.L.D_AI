import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict
import pandas as pd
from datetime import datetime

# Add project root to path to ensure correct imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.streaming.telemetry_validator import TelemetryValidator
from src.streaming.buffer import TelemetryBuffer
from src.streaming.rolling_stats import RollingStatisticsEngine
from src.streaming.derived_features import DerivedFeatureEngine
from src.streaming.online_store import OnlineFeatureStore
from src.streaming.stream_health import StreamHealthMonitor
from src.inference.engine import InferenceEngine

logger = logging.getLogger("telemetry_replay")


def normalize_recommendation(rec: Any) -> Dict[str, Any]:
    """Normalizes safety recommendations from different formats to a standard schema.
    
    Unified Schema:
    {
        "priority": str,      # Safety/severity level (e.g. Low, Medium, High, Critical)
        "action": str,        # Mitigation action
        "reason": str,        # Detail/reason for the recommendation
        "category": str       # Area or hazard category
    }
    """
    if not isinstance(rec, dict):
        raise ValueError(f"Recommendation must be a dictionary, got {type(rec).__name__}")

    # 1. Normalize Priority / Severity
    priority = rec.get("priority") or rec.get("severity") or "HIGH"
    priority = str(priority).strip()

    # 2. Normalize Action / Recommended Action
    action = rec.get("action") or rec.get("recommended_action")
    if not action:
        raise ValueError("Recommendation action field is missing or empty")
    action = str(action).strip()

    # 3. Normalize Reason / Description
    reason = rec.get("reason") or rec.get("description") or "Safety threshold limit exceeded"
    reason = str(reason).strip()

    # 4. Normalize Category / Hazard Category / Process Area
    category = rec.get("category") or rec.get("hazard_category") or rec.get("process_area") or "General"
    category = str(category).strip()

    return {
        "priority": priority,
        "action": action,
        "reason": reason,
        "category": category
    }


def print_header(title: str) -> None:
    print("\033[1;36m" + "=" * 60 + "\033[0m")
    print(f"\033[1;37m   {title.upper()}   \033[0m")
    print("\033[1;36m" + "=" * 60 + "\033[0m")


def get_color_class(pred_class: str) -> str:
    colors = {
        "Low": "\033[1;32mLow\033[0m",         # Green
        "Medium": "\033[1;33mMedium\033[0m",   # Yellow
        "High": "\033[1;31mHigh\033[0m",       # Light Red
        "Critical": "\033[41;1;37mCritical\033[0m" # Highlighted Red
    }
    return colors.get(pred_class, pred_class)


def print_dashboard(
    row_idx: int,
    raw_sample: dict,
    pred_res: dict,
    derived: dict,
    health: dict
) -> None:
    # Clear console (ANSI escape sequence)
    print("\033[H\033[J", end="")
    
    print_header("S.H.E.I.L.D. Streaming Intelligence Control Room")
    
    timestamp = raw_sample.get("Timestamp", "N/A")
    pred_class = pred_res.get("predicted_class", "N/A")
    confidence = pred_res.get("confidence", 0.0) * 100.0
    
    print(f"Timestamp: \033[1;35m{timestamp}\033[0m | Sample Index: {row_idx + 1}")
    print(f"Prediction: {get_color_class(pred_class)} (Confidence: \033[1;37m{confidence:.2f}%\033[0m)")
    print("-" * 60)
    
    # 1. Safety Indices
    print("\033[1;34mSAFETY & RISK INDEX MONITOR\033[0m")
    print(f"  Safety Index:        {derived.get('Safety_Index', 0.0):6.2f}  |  Process Stability:  {derived.get('Process_Stability_Index', 0.0):6.2f}")
    print(f"  Compound Risk Score: {derived.get('Compound_Risk_Score', 0.0):6.2f}  |  Gas Risk Index:     {derived.get('Gas_Risk_Index', 0.0):6.2f}")
    print(f"  Equipment Health:    {derived.get('Equipment_Health_Index', 0.0):6.2f}  |  Worker Exposure:    {derived.get('Worker_Exposure_Index', 0.0):6.2f}")
    print(f"  Thermal Risk Index:  {derived.get('Thermal_Risk_Index', 0.0):6.2f}  |  Maintenance Risk:   {derived.get('Maintenance_Risk_Index', 0.0):6.2f}")
    print("-" * 60)
    
    # 2. Pipeline Health
    status_str = health["status"]
    if status_str == "HEALTHY":
        status_colored = f"\033[1;32m{status_str}\033[0m"
    elif status_str == "WARNING":
        status_colored = f"\033[1;33m{status_str}\033[0m"
    else:
        status_colored = f"\033[1;31m{status_str}\033[0m"
        
    print(f"\033[1;34mPIPELINE HEALTH & THROUGHPUT\033[0m")
    print(f"  Status:             {status_colored}")
    print(f"  Ingestion Volume:   {health['packets_received']} packets")
    print(f"  Buffer Utilization: {health['buffer_utilization_pct']:.1f}% ({health['buffer_size']}/{health['buffer_capacity']})")
    print(f"  Packet Loss Rate:   {health['packet_loss_rate_pct']:.2f}%")
    print("-" * 60)
    
    # 3. Latency breakdown
    latencies = health["latest_latencies_ms"]
    avg_latencies = health["average_latencies_ms"]
    
    print(f"\033[1;34mLATENCY PROFILE (SLA Budget: {health['critical_threshold_ms']:.1f}ms)\033[0m")
    print(f"  Phase                     Latest Latency      Avg Latency (EMA)")
    print(f"  1. Validation             {latencies['validation']:10.3f} ms      {avg_latencies['validation']:10.3f} ms")
    print(f"  2. Rolling Stats          {latencies['rolling_stats']:10.3f} ms      {avg_latencies['rolling_stats']:10.3f} ms")
    print(f"  3. Derived Features       {latencies['derived_features']:10.3f} ms      {avg_latencies['derived_features']:10.3f} ms")
    print(f"  4. Feature Store          {latencies['feature_store']:10.3f} ms      {avg_latencies['feature_store']:10.3f} ms")
    print(f"  5. Model Inference        {latencies['inference']:10.3f} ms      {avg_latencies['inference']:10.3f} ms")
    print(f"  \033[1;37mTotal End-to-End Latency  {latencies['total_end_to_end']:10.3f} ms      {avg_latencies['total_end_to_end']:10.3f} ms\033[0m")
    print("-" * 60)
    
    # Recommendations (if any)
    recos = pred_res.get("recommendations", [])
    if recos:
        print("\033[1;31mOPERATIONAL RECOMMENDATIONS:\033[0m")
        for rec in recos[:3]:  # Top 3
            try:
                norm_rec = normalize_recommendation(rec)
                print(f"  * \033[1;33m[{norm_rec['priority'].upper()}]\033[0m: {norm_rec['action']}")
            except Exception as e:
                logger.warning("Skipping invalid recommendation: %s. Error: %s", str(rec), str(e))
            
    print("\nPress Ctrl+C to terminate the stream replay.")


def run_replay(csv_path: str, limit: int, realtime: bool, window_size: int = 30) -> None:
    # 1. Verify files exist
    if not os.path.exists(csv_path):
        print(f"Error: Telemetry CSV file not found at: {csv_path}")
        sys.exit(1)

    print("Initializing Streaming Safety Pipeline components...")
    
    # 2. Instantiate pipeline classes
    validator = TelemetryValidator()
    buffer = TelemetryBuffer(max_size=window_size)
    stats_engine = RollingStatisticsEngine(windows=[5, 15, 30])
    derived_engine = DerivedFeatureEngine()
    store = OnlineFeatureStore()
    monitor = StreamHealthMonitor(latency_critical_threshold_ms=200.0, latency_warning_threshold_ms=50.0)
    inference_engine = InferenceEngine()

    print("Loading telemetry dataset...")
    df_all = pd.read_csv(csv_path, keep_default_na=False)
    
    # Sort chronologically to simulate stream correctly
    df_all["ts_dt"] = pd.to_datetime(df_all["Timestamp"])
    df = df_all.sort_values("ts_dt").head(limit)
    
    print(f"Successfully loaded and sorted {len(df)} samples for replay.")
    time.sleep(1.0)

    raw_keys = validator.RAW_SENSORS.union(validator.CATEGORICAL_FIELDS.keys()).union(validator.BOOLEAN_FIELDS)

    try:
        for idx, row in df.reset_index(drop=True).iterrows():
            # Measure complete end-to-end streaming processing time
            start_end_to_end = time.perf_counter()
            raw_sample = row.to_dict()
            
            # Extract raw telemetry values
            raw_telemetry = {k: raw_sample[k] for k in raw_keys if k in raw_sample}
            raw_telemetry["Timestamp"] = raw_sample["Timestamp"]

            # Record packet arrival
            monitor.record_packet_received(timestamp=raw_sample["Timestamp"])
            
            # Step 1: Validate Telemetry
            t0 = time.perf_counter()
            is_valid, reason = validator.validate(raw_telemetry)
            validation_time = time.perf_counter() - t0
            monitor.record_latency("validation", validation_time)
            
            if not is_valid:
                monitor.record_validation_failure()
                if realtime:
                    print(f"\033[1;31m[ERROR] Packet failed validation: {reason}\033[0m")
                    time.sleep(0.5)
                continue

            # Normalize values
            normalized = validator.normalize(raw_telemetry)
            # Inject Safety_Index to mimic rule engine's pre-calculated output
            normalized["Safety_Index"] = raw_sample.get("Safety_Index", 100.0)
            
            # Step 2: Buffer Telemetry
            buffer.append(normalized)
            monitor.record_buffer_utilization(buffer.current_size(), buffer._max_size)

            # Step 3: Compute Rolling Statistics
            t0 = time.perf_counter()
            rolling_stats = stats_engine.update(buffer)
            rolling_time = time.perf_counter() - t0
            monitor.record_latency("rolling_stats", rolling_time)

            # Step 4: Compute Derived Safety Indices
            t0 = time.perf_counter()
            derived_features = derived_engine.compute(normalized, rolling_stats)
            derived_time = time.perf_counter() - t0
            monitor.record_latency("derived_features", derived_time)

            # Step 5: Online Feature Store mapping
            t0 = time.perf_counter()
            feature_vector = store.serve_feature_vector(normalized, rolling_stats, derived_features)
            store_time = time.perf_counter() - t0
            monitor.record_latency("feature_store", store_time)

            # Step 6: Model Inference Execution
            # Merge to pass to InferenceEngine (which expects raw + engineered columns for standard scaling)
            merged = {}
            merged.update(normalized)
            merged.update(rolling_stats)
            merged.update(derived_features)
            
            t0 = time.perf_counter()
            pred_res = inference_engine.predict_single(merged)
            inference_time = time.perf_counter() - t0
            monitor.record_latency("inference", inference_time)

            # End of Pipeline Step
            total_time = time.perf_counter() - start_end_to_end
            monitor.record_latency("total_end_to_end", total_time)

            # Dashboard reporting
            if realtime:
                print_dashboard(
                    row_idx=idx,
                    raw_sample=raw_sample,
                    pred_res=pred_res,
                    derived=derived_features,
                    health=monitor.get_metrics()
                )
                time.sleep(0.1)  # Sleep to simulate real-time ingestion rate
            else:
                # Periodic summary updates in batch mode
                if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
                    metrics = monitor.get_metrics()
                    print(f"Processed: {idx + 1}/{len(df)} | Status: {metrics['status']} | Avg Latency: {metrics['average_latencies_ms']['total_end_to_end']:.3f} ms | Packet Loss: {metrics['packet_loss_rate_pct']}%")

    except KeyboardInterrupt:
        print("\nReplay aborted by user.")

    # Print Final Health Summary Report
    metrics = monitor.get_metrics()
    print("\n" + "=" * 60)
    print("   STREAM REPLAY SUMMARY REPORT   ")
    print("=" * 60)
    print(f"Overall Status:       {metrics['status']}")
    print(f"Total Packets:        {metrics['packets_received']}")
    print(f"Failed Packets:       {metrics['packets_failed_validation']}")
    print(f"Packet Loss Rate:     {metrics['packet_loss_rate_pct']}%")
    print(f"Avg End-to-End Time:  {metrics['average_latencies_ms']['total_end_to_end']:.3f} ms")
    print(f"Avg Inference Time:   {metrics['average_latencies_ms']['inference']:.3f} ms")
    print(f"Avg Engine Time:      {(metrics['average_latencies_ms']['total_end_to_end'] - metrics['average_latencies_ms']['inference']):.3f} ms")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S.H.E.I.L.D. Telemetry Stream Replay Runner")
    parser.add_argument("--csv", type=str, default="data/industrial_safety_dataset_3.0.csv", help="Path to industrial dataset CSV")
    parser.add_argument("--limit", type=int, default=200, help="Number of rows to replay")
    parser.add_argument("--realtime", action="store_true", help="Run in interactive real-time dashboard mode")
    parser.add_argument("--window-size", type=int, default=30, help="Telemetry sliding buffer window size")
    args = parser.parse_args()

    run_replay(csv_path=args.csv, limit=args.limit, realtime=args.realtime, window_size=args.window_size)
