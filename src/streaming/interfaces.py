from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

class ITelemetryBuffer(ABC):
    """Abstract interface for sliding telemetry window buffers."""

    @abstractmethod
    def append(self, sample: Dict[str, Any]) -> None:
        """Appends a new normalized sample to the buffer."""
        pass

    @abstractmethod
    def get_window(self, size: int) -> List[Dict[str, Any]]:
        """Retrieves a historical window slice of the specified size.

        If fewer samples than size exist, returns the entire available buffer
        following the dynamic window startup strategy.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all buffered telemetry data."""
        pass

    @abstractmethod
    def current_size(self) -> int:
        """Returns the current number of items stored in the buffer."""
        pass

    @abstractmethod
    def is_ready(self, window: int) -> bool:
        """Checks if the buffer has accumulated enough samples for a given lookback window."""
        pass

    @abstractmethod
    def latest(self) -> Optional[Dict[str, Any]]:
        """Returns the most recently appended telemetry sample, if any."""
        pass

    @abstractmethod
    def oldest(self) -> Optional[Dict[str, Any]]:
        """Returns the oldest telemetry sample currently in the buffer, if any."""
        pass

    @abstractmethod
    def get_all(self) -> List[Dict[str, Any]]:
        """Returns all samples in the buffer ordered from oldest to newest."""
        pass


class IRollingStatsEngine(ABC):
    """Abstract interface for mathematical rolling computations."""

    @abstractmethod
    def update(self, buffer: ITelemetryBuffer) -> Dict[str, float]:
        """Runs incremental/windowed statistical computations on the buffered samples."""
        pass


class IDerivedFeatureEngine(ABC):
    """Abstract interface for domain-specific safety index calculations."""

    @abstractmethod
    def compute(self, telemetry: Dict[str, Any], rolling_stats: Dict[str, float]) -> Dict[str, float]:
        """Computes safety and risk indices using telemetry and mathematical statistics."""
        pass


class IOnlineFeatureStore(ABC):
    """Abstract interface for assembling and serving feature vectors."""

    @abstractmethod
    def serve_feature_vector(self, telemetry: Dict[str, Any], rolling_stats: Dict[str, float], derived_features: Dict[str, float]) -> Dict[str, Any]:
        """Aligns all raw and engineered inputs into the correct model schema (FeatureSet_B)."""
        pass

    @abstractmethod
    def get_schema(self) -> List[str]:
        """Returns the list of target features in the schema."""
        pass
