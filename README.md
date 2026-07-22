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

## Dataset

The training dataset is **not included** in this repository. The primary dataset file (`industrial_safety_dataset_3.0.csv`) is approximately **144 MB**, which exceeds [GitHub's 100 MB per-file size limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-and-github). To keep the repository lightweight and push-friendly, the dataset is hosted externally on Google Drive instead.

### Download Instructions

1. **Download the dataset** from Google Drive:

   [Download industrial_safety_dataset_3.0.csv](https://drive.google.com/file/d/19hi2bPWBVngs-eoYaZbl7TTLNzwHHGFk/view?usp=drive_link)

2. **Create the `data/` folder** in the project root if it does not already exist:

   ```bash
   mkdir data
   ```

3. **Move the downloaded file** into the `data/` directory so the path looks like this:

   ```text
   S.H.E.I.L.D/
   ├── data/
   │   └── industrial_safety_dataset_3.0.csv
   ├── src/
   ├── models/
   ├── README.md
   └── requirements.txt
   ```

That's it — no renaming or extra setup is required.

### Why the dataset is not in Git

The `data/` directory is listed in [`.gitignore`](.gitignore) and is intentionally excluded from version control. This is standard practice for machine learning projects: source code, configuration, and documentation live in Git, while large datasets and trained model artifacts stay on disk locally (or in external storage). This keeps the repository fast to clone, easy to browse on GitHub, and free of file-size push errors.

> **Note:** Pre-trained model files under `models/` are also excluded from Git for the same reason. Run `python run_pipeline.py` to train and generate them locally after downloading the dataset.

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository, install dependencies, and download the dataset (see [Dataset](#dataset) above):

```bash
git clone https://github.com/jai571771/S.H.E.I.L.D_AI.git
cd S.H.E.I.L.D_AI
pip install -r requirements.txt
```

Before running the training pipeline or full evaluation workflow, make sure `data/industrial_safety_dataset_3.0.csv` is in place.

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
S.H.E.I.L.D_AI/
├── app.py                      # Interactive Streamlit HMI Dashboard
├── telemetry_replay.py         # Real-time CLI SCADA telemetry stream runner
├── run_pipeline.py             # Model retraining & certification audit runner
├── manual.py                   # Single-sample manual inference utility
├── requirements.txt            # Python dependencies
├── configs/                    # Pipeline configuration & safety rules JSON
│   ├── config.yaml
│   └── safety_rules.json
├── data/                       # Training dataset (not in Git — see Dataset section)
├── models/                     # Serialized XGBoost model artifacts (not in Git)
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
