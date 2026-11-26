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
| **S24CSEU2331** | Alankrit Kasana | Model Lead & Ops Lead |
| **S24CSEU2304** | Agrim Aggarwal | Data Lead |
| **S24CSEU2337** | Parth Makkar | Service Lead |
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

### 4. Data Cleaning (Stage 2)
Clean raw training data by resolving anomalies:
```bash
python -m src.mp055.clean
```

### 5. Model Training & MLflow Tracking (Stage 2)
Train regression pipelines across candidate settings, log to SQLite MLflow, and register `@champion`:
```bash
python -m src.mp055.train
```

### 6. Running Tests
Execute the full test suite with pytest:
```bash
pytest
```

### 7. Running the FastAPI Service (Stage 3)
Start the inference server locally:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Test the health probe:
```bash
curl http://localhost:8000/health
```
Test prediction endpoint:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"worker_exp_yr": 8, "rain_mm_week": 135.0, "slope_deg": 16, "temp_c": 23.0, "bush_age_yr": 30, "hours_worked": 7.5, "flush": "first", "section": "Upper Garden"}'
```

### 8. Containerization with Docker (Stage 3)
Build and run the container:
```bash
docker build -t mp055-api:1.0 .
docker run -d -p 8000:8000 --name mp055-service mp055-api:1.0
```

### 9. Kubernetes Orchestration (Stage 3)
Deploy the 4-replica service to Kubernetes:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```
