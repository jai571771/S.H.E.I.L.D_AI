import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.utils import get_config, setup_logger, PROJECT_ROOT

logger = setup_logger("pipeline.recommendations")

class SafetyRecommendationEngine:
    """Evaluates telemetry data against the knowledge base safety rules and suggests mitigation steps."""

    def __init__(self, rules_json_path: Optional[str] = None) -> None:
        """Initializes the recommendation engine by loading safety_rules.json."""
        config = get_config()
        if rules_json_path is None:
            self.rules_path = PROJECT_ROOT / config["paths"]["rules_json_path"]
        else:
            self.rules_path = Path(rules_json_path)

        self.rules: List[Dict[str, Any]] = []
        self.load_rules()

    def load_rules(self) -> None:
        """Loads safety rules from the JSON catalog file."""
        if not self.rules_path.exists():
            logger.warning("Rules database not found at %s. No rules will be active.", self.rules_path)
            return

        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
            logger.info("Loaded %d safety rules from: %s", len(self.rules), self.rules_path.name)
        except Exception as e:
            logger.error("Failed to load safety rules: %s", str(e))
            self.rules = []

    def evaluate(self, telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluates a single telemetry sample against the active safety rules.

        Checks for pre-calculated 'Rule_R001' to 'Rule_R020' flags in the telemetry dictionary,
        and falls back to rule condition approximations if flags are absent.

        Args:
            telemetry: Raw sensor and context telemetry packet.

        Returns:
            A list of triggered rules with name, description, severity, and recommended action.
        """
        active_recommendations = []

        # 1. Primary check: check pre-computed binary flags (Rule_R001 to Rule_R020)
        has_flags = any(f"Rule_R{i:03d}" in telemetry for i in range(1, 21))
        
        if has_flags:
            for rule in self.rules:
                rule_id = rule["rule_id"]
                # Match exact column name in telemetry (e.g. 'Rule_R001')
                col_name = f"Rule_{rule_id}"
                
                # Check if it exists and evaluates to True/1
                val = telemetry.get(col_name)
                if val is not None:
                    # Convert to float/int and check
                    try:
                        is_triggered = float(val) >= 0.5
                    except (ValueError, TypeError):
                        is_triggered = str(val).strip().lower() in ("true", "1", "1.0", "yes")
                        
                    if is_triggered:
                        active_recommendations.append({
                            "rule_id": rule_id,
                            "rule_name": rule["rule_name"],
                            "process_area": rule["process_area"],
                            "hazard_category": rule["hazard_category"],
                            "description": rule["compound_risk_reason"],
                            "severity": rule["risk_level"],
                            "recommended_action": rule["recommended_action"]
                        })
        
        # 2. Fallback check: check basic manual limits if no Rule flags are found or as a redundancy
        else:
            # Let's write simple rule logic matching some critical boundaries
            # BF CO limit
            bf_co = float(telemetry.get("BF_CO", 0.0) or 0.0)
            if bf_co >= 3.0:
                # Trigger R004 "Worker Gas Exposure" if worker present
                worker_present = str(telemetry.get("Worker_Count", 0)) != "0"
                if worker_present:
                    r004 = next((r for r in self.rules if r["rule_id"] == "R004"), None)
                    if r004:
                        active_recommendations.append({
                            "rule_id": "R004",
                            "rule_name": r004["rule_name"],
                            "process_area": r004["process_area"],
                            "hazard_category": r004["hazard_category"],
                            "description": r004["compound_risk_reason"],
                            "severity": r004["risk_level"],
                            "recommended_action": r004["recommended_action"]
                        })
            
            # Coke Oven Gas Leak (R011) or CO Level Warning
            co_co = float(telemetry.get("CO_CO", 0.0) or 0.0)
            maint_active = str(telemetry.get("Maintenance_Active", "")).lower() in ("true", "1", "yes")
            if co_co >= 3.0 and maint_active:
                r011 = next((r for r in self.rules if r["rule_id"] == "R011"), None)
                if r011:
                    active_recommendations.append({
                        "rule_id": "R011",
                        "rule_name": r011["rule_name"],
                        "process_area": r011["process_area"],
                        "hazard_category": r011["hazard_category"],
                        "description": r011["compound_risk_reason"],
                        "severity": r011["risk_level"],
                        "recommended_action": r011["recommended_action"]
                    })
                    
            # NH3 Leak (R016)
            co_nh3 = float(telemetry.get("CO_NH3", 0.0) or 0.0)
            if co_nh3 >= 0.5:
                r016 = next((r for r in self.rules if r["rule_id"] == "R016"), None)
                if r016:
                    active_recommendations.append({
                        "rule_id": "R016",
                        "rule_name": r016["rule_name"],
                        "process_area": r016["process_area"],
                        "hazard_category": r016["hazard_category"],
                        "description": r016["compound_risk_reason"],
                        "severity": r016["risk_level"],
                        "recommended_action": r016["recommended_action"]
                    })

        return active_recommendations
