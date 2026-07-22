import logging
from typing import Dict, Any, List
from src.streaming.interfaces import IDerivedFeatureEngine
from src.utils import setup_logger

logger = setup_logger("streaming.derived_features")


class DerivedFeatureEngine(IDerivedFeatureEngine):
    """Computes industrial domain safety indicators and risk indices from telemetry and statistics."""

    PHYSICAL_SENSORS = [
        "BF_Blast_Flow", "BF_Top_Temperature", "CO_Gas_Pressure", "CO_CO", "BF_H2", "CO_NH3",
        "BF_Blower_Current", "BF_Blower_Vibration", "BF_Burden_Level", "BF_Cooling_Water_Flow",
        "BF_Cooling_Water_Temperature", "BF_Top_Pressure", "CO_Oven_Temperature", "CO_Flue_Temperature",
        "CO_Pusher_Current", "CO_Pusher_Vibration", "CO_Quenching_Water_Flow", "CO_Steam_Temperature"
    ]

    def __init__(self) -> None:
        """Initializes the DerivedFeatureEngine and its lightweight state."""
        # Lightweight history to compute safety index decay for lead indicator
        self._safety_index_history: List[float] = []
        logger.info("DerivedFeatureEngine initialized.")

    def compute(self, telemetry: Dict[str, Any], rolling_stats: Dict[str, float]) -> Dict[str, float]:
        """Computes domain-specific industrial safety and risk indices.

        Args:
            telemetry: Current normalized telemetry sample dictionary.
            rolling_stats: Latest computed rolling statistics dictionary.

        Returns:
            A dictionary containing all calculated derived indices.
        """
        derived = {}

        # 1. Process Stability Index
        grad_bf_press = rolling_stats.get("BF_Top_Pressure_roll_grad_5", 0.0)
        grad_bf_flow = rolling_stats.get("BF_Blast_Flow_roll_grad_5", 0.0)
        grad_co_press = rolling_stats.get("CO_Gas_Pressure_roll_grad_5", 0.0)
        grad_co_temp = rolling_stats.get("CO_Oven_Temperature_roll_grad_5", 0.0)

        norm_grad_bf_press = abs(grad_bf_press) / 2.0
        norm_grad_bf_flow = abs(grad_bf_flow) / 1000.0
        norm_grad_co_press = abs(grad_co_press) / 5.0
        norm_grad_co_temp = abs(grad_co_temp) / 200.0

        stability_loss = min(100.0, 200.0 * (norm_grad_bf_press + norm_grad_bf_flow + norm_grad_co_press + norm_grad_co_temp))
        derived["Process_Stability_Index"] = float(100.0 - stability_loss)

        # 2. Gas Risk Index
        bf_co = telemetry.get("BF_CO", 0.0)
        bf_h2 = telemetry.get("BF_H2", 0.0)
        co_co = telemetry.get("CO_CO", 0.0)
        co_nh3 = telemetry.get("CO_NH3", 0.0)

        risk_bf_co = min(100.0, max(0.0, (bf_co - 40.0) / (200.0 - 40.0) * 100.0))
        risk_bf_h2 = min(100.0, max(0.0, (bf_h2 - 3.0) / (10.0 - 3.0) * 100.0))
        risk_co_co = min(100.0, max(0.0, (co_co - 5.0) / (25.0 - 5.0) * 100.0))
        risk_co_nh3 = min(100.0, max(0.0, (co_nh3 - 3.0) / (15.0 - 3.0) * 100.0))

        gas_risk_index = max(risk_bf_co, risk_bf_h2, risk_co_co, risk_co_nh3)
        derived["Gas_Risk_Index"] = float(gas_risk_index)

        # 3. Equipment Health Index
        bf_vib = telemetry.get("BF_Blower_Vibration", 0.0)
        co_vib = telemetry.get("CO_Pusher_Vibration", 0.0)
        bf_temp_water = telemetry.get("BF_Cooling_Water_Temperature", 35.0)
        co_temp_flue = telemetry.get("CO_Flue_Temperature", 1200.0)

        stress_bf_vib = min(100.0, max(0.0, (bf_vib - 1.2) / (3.5 - 1.2) * 100.0))
        stress_co_vib = min(100.0, max(0.0, (co_vib - 1.8) / (4.0 - 1.8) * 100.0))
        stress_bf_temp = min(100.0, max(0.0, (bf_temp_water - 38.0) / (60.0 - 38.0) * 100.0))
        stress_co_temp = min(100.0, max(0.0, (co_temp_flue - 1220.0) / (1350.0 - 1220.0) * 100.0))

        max_stress = max(stress_bf_vib, stress_co_vib, stress_bf_temp, stress_co_temp)
        equipment_health_index = 100.0 - max_stress
        derived["Equipment_Health_Index"] = float(equipment_health_index)

        # 4. Worker Exposure Index
        worker_count = telemetry.get("Worker_Count", 0)
        ppe_compliance = telemetry.get("PPE_Compliance", True)
        cctv_event = telemetry.get("CCTV_Event", "Normal")

        if worker_count == 0:
            worker_exposure_index = 0.0
        else:
            worker_factor = min(40.0, max(0.0, worker_count / 10.0 * 40.0))
            ppe_factor = 30.0 if not ppe_compliance else 0.0
            cctv_factor = 30.0 if cctv_event in ["Worker No PPE", "Unauthorized Entry"] else 0.0

            bf_state_id = telemetry.get("BF_State_ID", "")
            is_instrument_failure = (bf_state_id == "PS022")
            gas_contrib = 0.0 if is_instrument_failure else (0.3 * gas_risk_index)

            worker_exposure_index = min(100.0, worker_factor + ppe_factor + cctv_factor + gas_contrib)
        derived["Worker_Exposure_Index"] = float(worker_exposure_index)

        # 5. Thermal Risk Index
        bf_top_temp = telemetry.get("BF_Top_Temperature", 200.0)
        co_oven_temp = telemetry.get("CO_Oven_Temperature", 1000.0)
        co_steam_temp = telemetry.get("CO_Steam_Temperature", 250.0)
        ambient_temp = telemetry.get("Ambient_Temp", 25.0)

        # Ignore false over-temperature spikes during instrument failure PS022
        bf_state_id = telemetry.get("BF_State_ID", "")
        if (bf_top_temp > 900.0 or co_oven_temp > 1400.0) or bf_state_id == "PS022":
            risk_bf_temp = 0.0
            risk_co_oven = 0.0
        else:
            risk_bf_temp = min(100.0, max(0.0, (bf_top_temp - 220.0) / (350.0 - 220.0) * 100.0))
            risk_co_oven = min(100.0, max(0.0, (co_oven_temp - 1100.0) / (1200.0 - 1100.0) * 100.0))

        risk_co_steam = min(100.0, max(0.0, (co_steam_temp - 240.0) / (320.0 - 240.0) * 100.0))
        risk_ambient = 100.0 if (ambient_temp > 42.0 or ambient_temp < 8.0) else 0.0

        thermal_risk_index = max(risk_bf_temp, risk_co_oven, risk_co_steam, risk_ambient)
        derived["Thermal_Risk_Index"] = float(thermal_risk_index)

        # 6. Maintenance Risk Index
        maintenance_active = telemetry.get("Maintenance_Active", False)
        permit_type = telemetry.get("Permit_Type", "None")
        gas_test_completed = telemetry.get("Gas_Test_Completed", True)
        shift_type = telemetry.get("Shift_Type", "Morning")

        if not maintenance_active:
            maintenance_risk_index = 0.0
        else:
            permit_factor = 30.0 if permit_type == "Hot Work" else (40.0 if permit_type == "Confined Space" else 0.0)
            gas_test_factor = 30.0 if not gas_test_completed else 0.0
            shift_factor = 10.0 if shift_type == "Night" else 0.0
            maintenance_risk_index = min(100.0, 30.0 + permit_factor + gas_test_factor + shift_factor)
        derived["Maintenance_Risk_Index"] = float(maintenance_risk_index)

        # 7. Compound Risk & Safety Index Calculations
        eq_risk = 100.0 - equipment_health_index
        risks = [eq_risk, worker_exposure_index, gas_risk_index, thermal_risk_index, maintenance_risk_index, stability_loss]
        max_risk = max(risks)
        other_risks_sum = sum(risks) - max_risk

        compound_risk_score = min(100.0, max_risk + 0.15 * other_risks_sum)
        derived["Compound_Risk_Score"] = float(compound_risk_score)

        # Respect safety index from raw features if provided, otherwise compute it
        safety_index = telemetry.get("Safety_Index")
        if safety_index is None:
            safety_index = float(100.0 - compound_risk_score)
        else:
            safety_index = float(safety_index)
        derived["Safety_Index"] = safety_index

        # Keep lightweight history of Safety Index values
        self._safety_index_history.append(safety_index)
        if len(self._safety_index_history) > 5:
            self._safety_index_history.pop(0)

        # 8. Data Quality Score
        out_of_bounds_count = 0
        bf_top_press_val = telemetry.get("BF_Top_Pressure", 2.0)
        if bf_top_press_val < 0.5 or bf_top_press_val > 3.5:
            out_of_bounds_count += 1
        bf_top_temp_val = telemetry.get("BF_Top_Temperature", 200.0)
        if bf_top_temp_val < 100.0 or bf_top_temp_val > 400.0:
            out_of_bounds_count += 1
        bf_vib_val = telemetry.get("BF_Blower_Vibration", 1.0)
        if bf_vib_val < 0.0 or bf_vib_val > 8.0:
            out_of_bounds_count += 1
        co_oven_temp_val = telemetry.get("CO_Oven_Temperature", 1000.0)
        if co_oven_temp_val < 800.0 or co_oven_temp_val > 1300.0:
            out_of_bounds_count += 1
        co_press_val = telemetry.get("CO_Gas_Pressure", 7.0)
        if co_press_val < 2.0 or co_press_val > 15.0:
            out_of_bounds_count += 1
        derived["Data_Quality_Score"] = float(max(0.0, 100.0 - 5.0 * out_of_bounds_count))

        # 9. Safety Lead Indicator
        safety_t = safety_index
        safety_t_minus_5 = self._safety_index_history[0] if self._safety_index_history else safety_t
        is_decaying = (safety_t - safety_t_minus_5) < -2.0

        high_gradient_detected = False
        for sensor in self.PHYSICAL_SENSORS:
            grad_5_key = f"{sensor}_roll_grad_5"
            grad_5_val = rolling_stats.get(grad_5_key, 0.0)
            sensor_val = telemetry.get(sensor, 0.0)
            if abs(grad_5_val) > 0.05 * abs(sensor_val):
                high_gradient_detected = True
                break

        lead_indicator = 1 if (is_decaying or (high_gradient_detected and safety_t < 95.0)) else 0
        derived["Safety_Lead_Indicator"] = float(lead_indicator)

        return derived

    def clear(self) -> None:
        """Resets safety history state."""
        self._safety_index_history.clear()
