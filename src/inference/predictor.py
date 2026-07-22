from typing import Dict, Any
from src.inference.engine import InferenceEngine

# Global singleton instance of InferenceEngine for performance efficiency
_engine = None

def predict(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Runs end-to-end real-time inference on a single SCADA sample.

    Args:
        sample: A dictionary of telemetry readings.

    Returns:
        A dictionary containing prediction class, confidence, probabilities, SHAP values, and recommendations.
    """
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine.predict_single(sample)
