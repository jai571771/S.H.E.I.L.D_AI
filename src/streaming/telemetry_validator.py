import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from src.utils import setup_logger

logger = setup_logger("streaming.telemetry_validator")


class TelemetryValidator:
    """Validates and normalizes raw streaming SCADA telemetry inputs."""

    # Authoritative schema definition
    RAW_SENSORS = {
        "BF_Blast_Flow", "BF_Top_Temperature", "CO_Gas_Pressure", "CO_CO", "BF_H2", "CO_NH3",
        "BF_Blower_Current", "BF_Blower_Vibration", "BF_Burden_Level", "BF_Cooling_Water_Flow",
        "BF_Cooling_Water_Temperature", "BF_Top_Pressure", "CO_Oven_Temperature", "CO_Flue_Temperature",
        "CO_Pusher_Current", "CO_Pusher_Vibration", "CO_Quenching_Water_Flow", "CO_Steam_Temperature",
        "Ambient_Temp", "Worker_Count", "BF_CO", "BF_Gas_Pressure"
    }

    CATEGORICAL_FIELDS = {
        "Shift_Type": {"Morning", "Afternoon", "Night"},
        "Permit_Type": {"None", "Hot Work", "Confined Space", "General"},
        "CCTV_Event": {"Normal", "Worker No PPE"}
    }

    BOOLEAN_FIELDS = {"PPE_Compliance", "Maintenance_Active", "Gas_Test_Completed"}

    # Typical physical bounds for sensors to flag extreme noise/errors
    SENSOR_BOUNDS = {
        "BF_Blast_Flow": (0.0, 5000.0),
        "BF_Top_Temperature": (0.0, 1000.0),
        "CO_Gas_Pressure": (0.0, 50.0),
        "CO_CO": (0.0, 100.0),
        "BF_H2": (0.0, 20.0),
        "CO_NH3": (0.0, 10.0),
        "BF_Blower_Current": (0.0, 10000.0),
        "BF_Blower_Vibration": (0.0, 20.0),
        "BF_Burden_Level": (0.0, 10.0),
        "BF_Cooling_Water_Flow": (0.0, 1000.0),
        "BF_Cooling_Water_Temperature": (0.0, 100.0),
        "BF_Top_Pressure": (0.0, 10.0),
        "CO_Oven_Temperature": (0.0, 2000.0),
        "CO_Flue_Temperature": (0.0, 2000.0),
        "CO_Pusher_Current": (0.0, 1000.0),
        "CO_Pusher_Vibration": (0.0, 20.0),
        "CO_Quenching_Water_Flow": (0.0, 1000.0),
        "CO_Steam_Temperature": (0.0, 500.0),
        "Ambient_Temp": (-40.0, 60.0),
        "Worker_Count": (0, 100),
        "BF_CO": (0.0, 100.0),
        "BF_Gas_Pressure": (0.0, 50.0)
    }

    # Operational defaults for missing values
    DEFAULTS = {
        "BF_Blast_Flow": 1500.0,
        "BF_Top_Temperature": 200.0,
        "CO_Gas_Pressure": 2.5,
        "CO_CO": 1.5,
        "BF_CO": 15.0,
        "BF_H2": 1.5,
        "CO_NH3": 0.1,
        "BF_Blower_Current": 3000.0,
        "BF_Blower_Vibration": 1.5,
        "BF_Burden_Level": 2.0,
        "BF_Cooling_Water_Flow": 150.0,
        "BF_Cooling_Water_Temperature": 35.0,
        "BF_Top_Pressure": 1.5,
        "CO_Oven_Temperature": 1000.0,
        "CO_Flue_Temperature": 1150.0,
        "CO_Pusher_Current": 150.0,
        "CO_Pusher_Vibration": 1.5,
        "CO_Quenching_Water_Flow": 100.0,
        "CO_Steam_Temperature": 250.0,
        "Ambient_Temp": 25.0,
        "Worker_Count": 5,
        "Shift_Type": "Morning",
        "PPE_Compliance": True,
        "Permit_Type": "None",
        "CCTV_Event": "Normal",
        "Maintenance_Active": False,
        "Gas_Test_Completed": True,
        "BF_Gas_Pressure": 2.0
    }

    def __init__(self) -> None:
        self.last_timestamp: Optional[datetime] = None

    def validate(self, telemetry: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validates schema, type, bounds, ordering, and categorical domains.

        Args:
            telemetry: Raw input telemetry dictionary.

        Returns:
            A tuple of (is_valid, error_reason_string).
        """
        # 1. Check Timestamp presence
        if "Timestamp" not in telemetry:
            return False, "Missing 'Timestamp' field."

        # 2. Validate Timestamp format and ordering
        try:
            ts_str = str(telemetry["Timestamp"])
            ts_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S") if len(ts_str) > 16 else datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
        except ValueError:
            try:
                ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError as e:
                return False, f"Invalid timestamp format: {ts_str}. Error: {str(e)}"

        if self.last_timestamp is not None and ts_dt <= self.last_timestamp:
            return False, f"Out-of-order or duplicate timestamp: {ts_str} <= latest {self.last_timestamp}."

        # 3. Check for presence of required variables
        missing_fields = []
        for key in self.RAW_SENSORS:
            if key not in telemetry:
                missing_fields.append(key)
        for key in self.CATEGORICAL_FIELDS:
            if key not in telemetry:
                missing_fields.append(key)
        for key in self.BOOLEAN_FIELDS:
            if key not in telemetry:
                missing_fields.append(key)

        if missing_fields:
            return False, f"Missing required telemetry fields: {missing_fields}"

        # 4. Check sensor bounds and float/int conversion
        for sensor, bounds in self.SENSOR_BOUNDS.items():
            val = telemetry[sensor]
            if val is None:
                continue
            try:
                f_val = float(val)
            except (ValueError, TypeError):
                return False, f"Field '{sensor}' value '{val}' cannot be converted to float."
            
            if f_val < bounds[0] or f_val > bounds[1]:
                logger.warning("Sensor value out of bounds: %s = %f (bounds: %s)", sensor, f_val, bounds)

        # 5. Check categorical domains
        for cat_field, allowed_values in self.CATEGORICAL_FIELDS.items():
            val = telemetry[cat_field]
            if val is not None and str(val) not in allowed_values:
                logger.warning("Unknown categorical value: %s = '%s' (expected one of %s)", cat_field, val, allowed_values)

        return True, None

    def normalize(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes field datatypes, fills missing values, and tracks timestamp state.

        Args:
            telemetry: Raw input telemetry dictionary.

        Returns:
            A cleaned and normalized telemetry packet dictionary.
        """
        normalized = {}
        
        # 1. Parse and format Timestamp
        ts_str = str(telemetry.get("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        try:
            ts_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S") if len(ts_str) > 16 else datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
        except ValueError:
            try:
                ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                ts_dt = datetime.now()
        
        self.last_timestamp = ts_dt
        normalized["Timestamp"] = ts_dt.strftime("%Y-%m-%d %H:%M:%S")

        # 2. Normalize and cast raw sensors
        for sensor in self.RAW_SENSORS:
            val = telemetry.get(sensor)
            if val is None or val == "":
                normalized[sensor] = self.DEFAULTS[sensor]
            else:
                try:
                    normalized[sensor] = float(val)
                except (ValueError, TypeError):
                    normalized[sensor] = self.DEFAULTS[sensor]

        # 3. Normalize categorical values
        for cat_field, allowed_values in self.CATEGORICAL_FIELDS.items():
            val = telemetry.get(cat_field)
            if val is None or val == "" or str(val) not in allowed_values:
                normalized[cat_field] = self.DEFAULTS[cat_field]
            else:
                normalized[cat_field] = str(val)

        # 4. Normalize booleans (maps Yes/No, True/False, and 1/0)
        for bool_field in self.BOOLEAN_FIELDS:
            val = telemetry.get(bool_field)
            if val is None or val == "":
                normalized[bool_field] = self.DEFAULTS[bool_field]
            elif isinstance(val, bool):
                normalized[bool_field] = val
            elif str(val).lower() in ("true", "yes", "1", "t", "y"):
                normalized[bool_field] = True
            elif str(val).lower() in ("false", "no", "0", "f", "n"):
                normalized[bool_field] = False
            else:
                normalized[bool_field] = self.DEFAULTS[bool_field]

        return normalized
