import logging
import math
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.streaming.interfaces import IRollingStatsEngine, ITelemetryBuffer
from src.utils import setup_logger

logger = setup_logger("streaming.rolling_stats")


class RollingStatisticsEngine(IRollingStatsEngine):
    """Calculates mathematical rolling window features incrementally from telemetry history."""

    def __init__(self, windows: List[int] = [5, 15, 30], config_dir: Optional[str] = None) -> None:
        """Initializes the RollingStatisticsEngine.

        Args:
            windows: Lookback window sizes in minutes/steps.
            config_dir: Optional directory containing pipeline configuration.
        """
        self.windows = windows
        self._last_stats: Dict[str, float] = {}
        
        # Determine physical sensors dynamically from config
        self.physical_sensors = self._load_physical_sensors(config_dir)
        
        # Running sum state: {sensor: {window: float}}
        self._running_sums: Dict[str, Dict[int, float]] = {s: {w: 0.0 for w in windows} for s in self.physical_sensors}
        # Running sum of squares state: {sensor: {window: float}}
        self._running_sum_sqs: Dict[str, Dict[int, float]] = {s: {w: 0.0 for w in windows} for s in self.physical_sensors}
        # Counter to reset accumulator drift
        self._step_counter = 0
        logger.info("RollingStatisticsEngine initialized dynamically with %d sensors and windows: %s", len(self.physical_sensors), windows)

    def _load_physical_sensors(self, config_dir: Optional[str]) -> List[str]:
        """Loads numeric columns from pipeline config and filters out calculated metrics/contexts."""
        from src.utils import PROJECT_ROOT, get_config
        import json
        
        default_sensors = [
            "BF_Blast_Flow", "BF_Top_Temperature", "CO_Gas_Pressure", "CO_CO", "BF_H2", "CO_NH3",
            "BF_Blower_Current", "BF_Blower_Vibration", "BF_Burden_Level", "BF_Cooling_Water_Flow",
            "BF_Cooling_Water_Temperature", "BF_Top_Pressure", "CO_Oven_Temperature", "CO_Flue_Temperature",
            "CO_Pusher_Current", "CO_Pusher_Vibration", "CO_Quenching_Water_Flow", "CO_Steam_Temperature"
        ]
        
        try:
            config = get_config()
            models_path = Path(config_dir) if config_dir else PROJECT_ROOT / config["paths"]["models_dir"]
            config_path = models_path / "pipeline_config.json"
            
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                num_cols = meta.get("numeric_columns", [])
                
                # Filter out derived features, rolling averages/gradients, ambient temp, and worker count
                filtered = []
                for col in num_cols:
                    is_derived = any(term in col for term in ["_roll_", "_Index", "_Score", "_Indicator", "_Quality"])
                    is_excluded = col in ["Worker_Count", "Ambient_Temp"]
                    if not is_derived and not is_excluded:
                        filtered.append(col)
                if filtered:
                    return filtered
        except Exception as e:
            logger.warning("Could not dynamically load physical sensors from config: %s. Using default list.", str(e))
            
        return default_sensors

    def update(self, buffer: ITelemetryBuffer) -> Dict[str, float]:
        """Calculates and updates rolling statistics from the telemetry buffer.

        Args:
            buffer: The active TelemetryBuffer.

        Returns:
            A dictionary containing rolling averages, variances, std devs, gradients,
            and rates of change for all physical sensors and windows.
        """
        self._step_counter += 1
        stats: Dict[str, float] = {}
        
        n = buffer.current_size()
        if n == 0:
            self._last_stats = stats
            return stats

        latest_sample = buffer.latest()
        if latest_sample is None:
            self._last_stats = stats
            return stats

        # Re-sync running state periodically to prevent floating-point accumulation drift
        if self._step_counter % 100 == 0:
            self._recalculate_accumulators(buffer)

        for sensor in self.physical_sensors:
            x_t = latest_sample.get(sensor, 0.0)
            
            for W in self.windows:
                effective_W = min(n, W)
                
                # 1. Update sums and sum of squares incrementally
                if n <= W:
                    # Warming up: just add the latest value, no value leaves the window
                    self._running_sums[sensor][W] += x_t
                    self._running_sum_sqs[sensor][W] += x_t * x_t
                else:
                    # Window is full: add the new value, subtract the leaving value
                    # The leaving value is the one that was at index n - W - 1
                    all_history = buffer.get_all()
                    x_leaving = all_history[n - W - 1].get(sensor, 0.0)
                    
                    self._running_sums[sensor][W] += x_t - x_leaving
                    self._running_sum_sqs[sensor][W] += (x_t * x_t) - (x_leaving * x_leaving)

                # Ensure values do not drop below zero due to precision errors
                curr_sum = self._running_sums[sensor][W]
                curr_sum_sq = max(0.0, self._running_sum_sqs[sensor][W])

                # 2. Compute Mean (Rolling Average)
                mean_val = curr_sum / effective_W
                stats[f"{sensor}_roll_avg_{W}"] = mean_val

                # 3. Compute Variance and Standard Deviation
                var_val = max(0.0, (curr_sum_sq / effective_W) - (mean_val * mean_val))
                stats[f"{sensor}_roll_var_{W}"] = var_val
                stats[f"{sensor}_roll_std_{W}"] = math.sqrt(var_val)

                # 4. Compute Rolling Gradient (exact match: (current_val - slice_vals[0]) / effective_W)
                all_history = buffer.get_all()
                x_start = all_history[n - effective_W].get(sensor, 0.0)
                
                if effective_W > 1:
                    grad_val = (x_t - x_start) / float(effective_W)
                    roc_val = (x_t - x_start) / (x_start + 1e-6)
                else:
                    grad_val = 0.0
                    roc_val = 0.0
                    
                stats[f"{sensor}_roll_grad_{W}"] = grad_val
                stats[f"{sensor}_roll_roc_{W}"] = roc_val

        self._last_stats = stats
        return stats

    def snapshot(self) -> Dict[str, float]:
        """Exposes a snapshot of the latest computed rolling statistics dictionary.

        Returns:
            A dictionary containing the latest statistics.
        """
        return self._last_stats.copy()

    def _recalculate_accumulators(self, buffer: ITelemetryBuffer) -> None:
        """Helper to recalculate sum accumulators from scratch to eliminate drift."""
        all_history = buffer.get_all()
        n = len(all_history)
        
        for sensor in self.physical_sensors:
            for W in self.windows:
                effective_W = min(n, W)
                slice_samples = all_history[-effective_W:]
                
                sum_val = 0.0
                sum_sq_val = 0.0
                for s in slice_samples:
                    val = s.get(sensor, 0.0)
                    sum_val += val
                    sum_sq_val += val * val
                    
                self._running_sums[sensor][W] = sum_val
                self._running_sum_sqs[sensor][W] = sum_sq_val


