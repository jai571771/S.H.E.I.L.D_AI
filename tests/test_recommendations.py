import unittest
from src.recommendations import SafetyRecommendationEngine

class TestSafetyRecommendationEngine(unittest.TestCase):
    """Tests the SafetyRecommendationEngine class."""

    def test_evaluate_rule_flags(self) -> None:
        engine = SafetyRecommendationEngine()
        
        # Test case 1: Telemetry with pre-computed rule flag (e.g. Rule_R001 = 1)
        telemetry_with_flags = {
            "BF_CO": 2.5,
            "Maintenance_Active": True,
            "Rule_R001": 1
        }
        recos = engine.evaluate(telemetry_with_flags)
        
        # Rule R001 should be triggered
        r001_triggered = any(r["rule_id"] == "R001" for r in recos)
        self.assertTrue(r001_triggered)
        
        # Test case 2: Telemetry with fallback conditions (e.g. CO_NH3 > 0.5)
        # We ensure no Rule_Rxxx flags are present
        telemetry_fallback = {
            "CO_NH3": 0.8,
            "Worker_Count": 3
        }
        recos_fallback = engine.evaluate(telemetry_fallback)
        r016_triggered = any(r["rule_id"] == "R016" for r in recos_fallback)
        self.assertTrue(r016_triggered)

if __name__ == "__main__":
    unittest.main()
