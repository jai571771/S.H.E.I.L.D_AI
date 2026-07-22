import http.server
import json
import logging
import sys
import urllib.parse
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_config, setup_logger
from src.inference import InferenceEngine

# Setup logging
logger = setup_logger("inference_server")

# Global data structures
df_raw = None
stream_index = 0
engine = None


def load_dataset():
    """Loads the raw generated dataset for streaming simulation."""
    global df_raw
    config = get_config()
    dataset_path = PROJECT_ROOT / config["paths"]["dataset_path"]
    
    if not dataset_path.exists():
        logger.error("Dataset not found at %s. Please run dataset generators first.", dataset_path)
        # Create a dummy dataframe for safety fallback
        df_raw = pd.DataFrame([{
            "Timestamp": "2026-07-17 09:00:00",
            "BF_Blast_Flow": 1500.0,
            "BF_Top_Temperature": 120.0,
            "CO_Gas_Pressure": 2.2,
            "CO_CO": 1.5,
            "BF_CO": 0.8,
            "BF_H2": 1.1,
            "CO_NH3": 0.05,
            "BF_Blower_Vibration": 1.2,
            "Worker_Count": 4,
            "PPE_Compliance": "Yes",
            "Permit_Type": "None",
            "CCTV_Event": "None",
            "Maintenance_Active": False,
            "Gas_Test_Completed": True,
            "Shift_Type": "Day",
            "Future_Risk_Level": "Low"
        }])
        return

    logger.info("Loading raw dataset for streaming simulation: %s...", dataset_path.name)
    try:
        df_raw = pd.read_csv(dataset_path)
        logger.info("Loaded %d rows for streaming simulation.", len(df_raw))
    except Exception as e:
        logger.error("Error loading raw dataset: %s", str(e))
        df_raw = pd.DataFrame()


class InferenceHandler(http.server.BaseHTTPRequestHandler):
    """Handles HTTP requests for static dashboard files and API endpoints."""

    def log_message(self, format, *args):
        # Override to suppress noisy request logging in terminal
        return

    def do_OPTIONS(self):
        """Allows CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handles GET requests for the dashboard and streaming telemetry."""
        global df_raw, stream_index

        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # 1. API Endpoint: GET /api/stream
        if path == "/api/stream":
            if df_raw is None or df_raw.empty:
                self.send_error_response("Dataset not initialized or empty", 500)
                return

            # Retrieve current row
            row = df_raw.iloc[stream_index].to_dict()
            stream_index = (stream_index + 1) % len(df_raw)

            # Convert numpy types to native Python types for JSON compatibility
            serializable_row = {}
            for k, v in row.items():
                if isinstance(v, (np.integer, np.int64)):
                    serializable_row[k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    serializable_row[k] = float(v)
                elif pd.isna(v):
                    serializable_row[k] = None
                else:
                    serializable_row[k] = v

            self.send_json_response(serializable_row)
            return

        # 2. Serve static dashboard page: GET / or index.html
        elif path == "/" or path == "/index.html":
            static_file = PROJECT_ROOT / "src" / "static" / "index.html"
            if not static_file.exists():
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Static dashboard page not found. Please verify src/static/index.html exists.")
                return

            try:
                with open(static_file, "r", encoding="utf-8") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Internal Server Error: {str(e)}".encode("utf-8"))
            return

        # 3. Not Found
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        """Handles POST requests for live risk classification."""
        global engine

        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # API Endpoint: POST /api/predict
        if path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                raw_telemetry = json.loads(post_data.decode("utf-8"))
            except Exception as e:
                self.send_error_response(f"Invalid JSON payload: {str(e)}", 400)
                return

            try:
                # Run real-time inference prediction
                prediction_result = engine.predict_single(raw_telemetry)
                self.send_json_response(prediction_result)
            except Exception as e:
                logger.error("Inference prediction failed: %s", str(e))
                self.send_error_response(f"Inference error: {str(e)}", 500)
            return

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def send_json_response(self, data, status=200):
        """Sends a structured JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def send_error_response(self, message, status=400):
        """Sends a structured error response."""
        response = {"success": False, "error": message}
        self.send_json_response(response, status)


def run_server(port=8000):
    """Starts the HTTP server."""
    global engine

    # Load dataset
    load_dataset()

    # Load Inference Engine
    try:
        engine = InferenceEngine()
    except Exception as e:
        logger.error("Failed to initialize InferenceEngine: %s. Please run model retraining/certification first.", str(e))
        sys.exit(1)

    server_address = ("", port)
    httpd = http.server.HTTPServer(server_address, InferenceHandler)
    logger.info("==========================================")
    logger.info("INDUSTRIAL SAFETY INFERENCE SERVER RUNNING")
    logger.info("Dashboard URL: http://localhost:%d", port)
    logger.info("API Endpoint:  http://localhost:%d/api/predict", port)
    logger.info("==========================================")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down inference server...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
