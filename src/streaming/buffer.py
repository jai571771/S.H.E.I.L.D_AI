import logging
from collections import deque
from typing import Dict, List, Any, Optional
from src.streaming.interfaces import ITelemetryBuffer
from src.utils import setup_logger

logger = setup_logger("streaming.buffer")


class TelemetryBuffer(ITelemetryBuffer):
    """Circular memory queue representing the lookback timeline for streaming data."""

    def __init__(self, max_size: Optional[int] = None) -> None:
        """Initializes the TelemetryBuffer with a config-driven size.

        Args:
            max_size: Optional manual override for maximum size.
        """
        if max_size is None:
            try:
                config = get_config()
                streaming_conf = config.get("streaming", {})
                lookback = float(streaming_conf.get("lookback_window_minutes", 30))
                interval = float(streaming_conf.get("sampling_interval_minutes", 1))
                self._max_size = int(lookback / interval)
            except Exception:
                # Default safety fallback
                self._max_size = 30
        else:
            self._max_size = max_size

        self._queue: deque = deque(maxlen=self._max_size)
        logger.info("TelemetryBuffer initialized with max lookback limit: %d", self._max_size)

    def append(self, sample: Dict[str, Any]) -> None:
        """Appends a new validated sample to the sliding window memory queue.

        Args:
            sample: Cleaned, normalized telemetry sample dictionary.
        """
        self._queue.append(sample)
        logger.debug("Telemetry buffer size: %d/%d", len(self._queue), self._max_size)

    def get_window(self, size: int) -> List[Dict[str, Any]]:
        """Retrieves a historical window slice of the specified size.

        Implements the dynamic rolling window startup strategy: if fewer samples
        than size are present, it returns all available samples.

        Args:
            size: Target window lookback length.

        Returns:
            A list of dictionary samples representing the historical timeline,
            sorted from oldest to newest.
        """
        curr_len = len(self._queue)
        effective_window = min(curr_len, size)
        
        if effective_window <= 0:
            return []
            
        full_history = list(self._queue)
        return full_history[-effective_window:]

    def clear(self) -> None:
        """Clears all buffered telemetry data."""
        self._queue.clear()
        logger.info("TelemetryBuffer has been cleared.")

    def current_size(self) -> int:
        """Returns the current size of the buffer."""
        return len(self._queue)

    def is_ready(self, window: int) -> bool:
        """Checks if the buffer has accumulated at least the specified window size.

        Args:
            window: Target number of samples.

        Returns:
            True if buffer size >= window, False otherwise.
        """
        return len(self._queue) >= window

    def latest(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent telemetry sample in the buffer.

        Returns:
            The latest dictionary sample, or None if the buffer is empty.
        """
        if not self._queue:
            return None
        return self._queue[-1]

    def oldest(self) -> Optional[Dict[str, Any]]:
        """Returns the oldest telemetry sample currently in the buffer.

        Returns:
            The oldest dictionary sample, or None if the buffer is empty.
        """
        if not self._queue:
            return None
        return self._queue[0]

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns all samples in the buffer ordered from oldest to newest.

        Returns:
            List of all telemetry samples.
        """
        return list(self._queue)
