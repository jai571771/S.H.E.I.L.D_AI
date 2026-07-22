# 🛡️ S.H.E.I.L.D. — Safety Hazard Evaluation & Intelligent Learning Dashboard

**S.H.E.I.L.D.** is a production-grade, real-time industrial HMI safety dashboard and telemetry intelligence platform. Powered by XGBoost machine learning models and SHAP explainability, S.H.E.I.L.D. provides live hazard evaluation, streaming telemetry processing, and process safety diagnostics for Blast Furnace & Coke Oven steel mill operations.

---

## 🌟 Key Features

*   **Real-Time HMI Safety Console**: Interactive glassmorphic Streamlit web application with live safety risk indicators, parameter gauges, and telemetry trend sparklines.
*   **Predictive AI Safety Classifier**: Pre-trained XGBoost model for predicting future hazard risk levels (`Low`, `Medium`, `High`, `Critical`).
*   **Explainable AI (SHAP)**: Dynamic feature attribution charts and automated natural language reasoning explaining top risk factors for every prediction.
*   **Streaming SCADA Telemetry Engine**: Packet-by-packet stream validator, sliding buffer, rolling statistics engine, and real-time SLA latency profiling.
*   **Operational Recommendations Engine**: Actionable mitigation guidance and safety rules for plant engineers based on real-time sensor limits and AI risk scores.

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/jai571771/S.H.I.E.L.D_AI.git
cd S.H.I.E.L.D_AI
pip install -r requirements.txt
```

---

### 2. Running the Interactive Streamlit Web Dashboard

Launch the live Streamlit HMI control room:

```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

### 3. Running Terminal SCADA Telemetry Simulator

Run the interactive terminal streaming control room:

```bash
# Real-Time Interactive Streaming Console
python telemetry_replay.py --realtime

# Batch Replay Mode
python telemetry_replay.py --limit 500
```

---

### 4. Running Single-Sample Manual Inference Test

Test custom telemetry inputs against the inference pipeline:

```bash
python manual.py
```

---

### 5. AI Pipeline Retraining & Certification Audit

To run full data validation, leakage removal, model training, benchmarking, and test set evaluation:

```bash
python run_pipeline.py
```

---

### 6. Automated Unit Tests

Run test suite:

```bash
pytest
```

---

## 📁 Repository Structure

```
S.H.I.E.L.D_AI/
├── app.py                      # Interactive Streamlit HMI Dashboard
├── telemetry_replay.py         # Real-time CLI SCADA telemetry stream runner
├── run_pipeline.py             # Model retraining & certification audit runner
├── manual.py                   # Single-sample manual inference utility
├── requirements.txt            # Python dependencies
├── configs/                    # Pipeline configuration & safety rules JSON
│   ├── config.yaml
│   └── safety_rules.json
├── data/                       # Industrial dataset CSVs
├── models/                     # Serialized XGBoost model artifacts
├── src/                        # Core backend modules
│   ├── streaming/              # Streaming pipeline (Validator, Buffer, Stats, Health)
│   ├── inference/              # Model inference engine & predictors
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   ├── trainer.py
│   ├── evaluation.py
│   └── explainability.py
├── reports/                    # Benchmark reports & evaluation metrics
└── tests/                      # Pytest unit & integration test suite
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
