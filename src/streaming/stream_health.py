import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("streaming.stream_health")


class StreamHealthMonitor:
    """Monitors the operational health, throughput, latency, and reliability of the streaming pipeline."""

    def __init__(self, latency_critical_threshold_ms: float = 200.0, latency_warning_threshold_ms: float = 50.0) -> None:
        """Initializes the StreamHealthMonitor.

        Args:
            latency_critical_threshold_ms: Total latency limit (e.g., 200ms SLA budget).
            latency_warning_threshold_ms: Internal pipeline goal (e.g., 50ms warning trigger).
        """
        self.critical_threshold = latency_critical_threshold_ms
        self.warning_threshold = latency_warning_threshold_ms
        
        # Counters
        self.packets_received = 0
        self.packets_failed_validation = 0
        self.last_packet_arrival_time: Optional[float] = None
        self.last_packet_timestamp_str: Optional[str] = None
        
        # Latency records (in milliseconds)
        self.latencies: Dict[str, float] = {
            "validation": 0.0,
            "rolling_stats": 0.0,
            "derived_features": 0.0,
            "feature_store": 0.0,
            "inference": 0.0,
            "total_end_to_end": 0.0
        }
        
        # Rolling averages for metrics
        self.avg_latencies: Dict[str, float] = {k: 0.0 for k in self.latencies}
        self.alpha = 0.1  # Exponential moving average factor
        
        # Buffer metrics
        self.buffer_size = 0
        self.buffer_capacity = 30
        
        logger.info("StreamHealthMonitor initialized (Warning: %.1fms, Critical: %.1fms).", 
                    self.warning_threshold, self.critical_threshold)

    def record_packet_received(self, timestamp: Optional[str] = None) -> None:
        """Records the arrival of a new telemetry packet."""
        self.packets_received += 1
        self.last_packet_arrival_time = time.time()
        if timestamp:
            self.last_packet_timestamp_str = timestamp

    def record_validation_failure(self) -> None:
        """Records a validation failure/dropped packet."""
        self.packets_failed_validation += 1

    def record_latency(self, phase: str, duration_seconds: float) -> None:
        """Records execution latency for a pipeline phase.

        Args:
            phase: Phase name (e.g., validation, rolling_stats, derived_features, feature_store, inference, total_end_to_end).
            duration_seconds: Latency in seconds.
        """
        duration_ms = duration_seconds * 1000.0
        if phase in self.latencies:
            self.latencies[phase] = duration_ms
            # Update EMA (Exponential Moving Average)
            self.avg_latencies[phase] = (self.alpha * duration_ms) + ((1.0 - self.alpha) * self.avg_latencies[phase])

    def record_buffer_utilization(self, current_size: int, capacity: int) -> None:
        """Updates buffer size and capacity utilization metrics."""
        self.buffer_size = current_size
        self.buffer_capacity = max(1, capacity)

    def get_metrics(self) -> Dict[str, Any]:
        """Exposes a snapshot of health, reliability, and latency metrics.

        Returns:
            A dictionary containing current metrics values.
        """
        utilization = (self.buffer_size / self.buffer_capacity) * 100.0 if self.buffer_capacity > 0 else 0.0
        loss_rate = (self.packets_failed_validation / self.packets_received * 100.0) if self.packets_received > 0 else 0.0
        
        # Calculate time since last packet
        time_since_last_packet = None
        if self.last_packet_arrival_time is not None:
            time_since_last_packet = time.time() - self.last_packet_arrival_time

        return {
            "status": self.check_health(),
            "packets_received": self.packets_received,
            "packets_failed_validation": self.packets_failed_validation,
            "packet_loss_rate_pct": round(loss_rate, 2),
            "time_since_last_packet_seconds": round(time_since_last_packet, 3) if time_since_last_packet is not None else None,
            "last_packet_timestamp": self.last_packet_timestamp_str,
            "buffer_utilization_pct": round(utilization, 2),
            "buffer_size": self.buffer_size,
            "buffer_capacity": self.buffer_capacity,
            "latest_latencies_ms": {k: round(v, 3) for k, v in self.latencies.items()},
            "average_latencies_ms": {k: round(v, 3) for k, v in self.avg_latencies.items()},
            "critical_threshold_ms": self.critical_threshold,
            "warning_threshold_ms": self.warning_threshold
        }

    def check_health(self) -> str:
        """Determines overall pipeline status based on latencies and packet loss.

        Returns:
            One of: "HEALTHY", "WARNING", "DEGRADED", "CRITICAL".
        """
        total_latency = self.latencies.get("total_end_to_end", 0.0)
        loss_rate = (self.packets_failed_validation / self.packets_received) if self.packets_received > 0 else 0.0
        
        # Check time since last packet (possible ingestion stall)
        if self.last_packet_arrival_time is not None:
            if (time.time() - self.last_packet_arrival_time) > 10.0:  # No packet for 10s
                return "DEGRADED"

        if total_latency >= self.critical_threshold or loss_rate >= 0.10:
            return "CRITICAL"
        elif total_latency >= self.warning_threshold or loss_rate >= 0.02:
            return "WARNING"
        
        return "HEALTHY"
