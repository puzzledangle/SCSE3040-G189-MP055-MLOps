# SCSE3040 MLOps Mini Project — Group G189

## Project Details
- **Problem ID:** MP055
- **Problem Title:** Tea Leaf Plucking (Indore)
- **Client:** Indore Tea Growers' Estate
- **Faculty:** Dr. Richa Sharma
- **Batch:** 87
- **Target Variable:** `kg_per_worker` (Regression, Evaluation Metric: MAE)

## Group Members
| Enrolment No. | Name | Role |
| :--- | :--- | :--- |
| **S24CSEU2337** | Parth Makkar | Data Lead |
| **S24CSEU2304** | Agrim Aggarwal | Model Lead |
| **S24CSEU2331** | Alankrit Kasana | Ops Lead & Service Lead |
| **S24CSEU2298** | HIMANSHI. | Member |

## Project Description
Indore Tea Growers' Estate pays workers by the kilogram of tea leaf plucked in a day and plans factory processing shifts based on expected yield. This project builds a production-grade MLOps pipeline to forecast daily plucked leaf weight (`kg_per_worker`) per worker using historical environmental and field operational features. The end-to-end architecture covers data version control (DVC), exploratory data validation and cleaning for real-world field data anomalies, tracked experimentation and model registry via MLflow, containerized REST API inference using FastAPI with input boundary validation, CI/CD automated testing, Kubernetes orchestration manifests, and post-deployment data drift monitoring (PSI) and model retraining.

## How to Run

### 1. Environment Setup
Create a Python 3.12 virtual environment and install the exact pinned dependencies:
```bash
# Windows (PowerShell)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Version Control (DVC)
Verify raw data tracking:
```bash
dvc status
```

### 3. Running Data Issue Diagnostics
Execute the Stage 1 data checks on raw training data:
```bash
python -m src.mp055.run_checks
```
