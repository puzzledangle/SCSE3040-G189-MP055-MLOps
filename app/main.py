"""FastAPI prediction service for MP055 (Tea Leaf Plucking Indore).

Provides:
- GET /health: Health probe endpoint for Kubernetes liveness/readiness
- POST /predict: Production inference endpoint with Pydantic validation
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, status
import joblib
import pandas as pd

from app.logger import setup_json_logger
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.mp055.clean import clean_flush_variants

logger = setup_json_logger()

# Global champion model pipeline reference
model_pipeline: Any = None
MODEL_PATH = Path("models/model.joblib")


def load_champion_model() -> Any:
    """Load the trained champion model pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Champion model not found at {MODEL_PATH.resolve()}. "
            "Execute 'python -m src.mp055.train' first."
        )
    return joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load model into memory on startup."""
    global model_pipeline
    try:
        model_pipeline = load_champion_model()
        logger.info("Champion model loaded successfully from models/model.joblib")
    except Exception as exc:
        logger.error(f"Failed to load model during startup: {exc}")
        model_pipeline = None
    yield


app = FastAPI(
    title="MP055 Tea Yield Prediction API",
    description="Forecasts daily tea leaf yield per worker (kg_per_worker) for Indore Tea Growers' Estate.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check probe",
)
def health_check() -> Dict[str, str]:
    """Health check endpoint used by Kubernetes readiness and liveness probes."""
    return {"status": "ok", "model": "mp055-model", "version": "champion"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict daily leaf yield per worker",
)
def predict_yield(payload: PredictionRequest) -> PredictionResponse:
    """Predict kg_per_worker from environmental and operational inputs."""
    global model_pipeline
    if model_pipeline is None:
        try:
            model_pipeline = load_champion_model()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model artifact not loaded: {exc}",
            )

    # Convert request payload to DataFrame with proper cleaning
    data_dict = payload.model_dump()
    df_input = pd.DataFrame([data_dict])
    df_input["flush"] = clean_flush_variants(df_input["flush"])

    try:
        preds = model_pipeline.predict(df_input)
        prediction_val = float(preds[0])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {exc}",
        )

    # Log structured JSON event to stdout (no PII)
    extra_record = {
        "prediction_kg": round(prediction_val, 2),
        "inputs": {
            "worker_exp_yr": payload.worker_exp_yr,
            "rain_mm_week": payload.rain_mm_week,
            "temp_c": payload.temp_c,
            "hours_worked": payload.hours_worked,
            "section": payload.section,
        },
    }
    logger.info("Inference completed", extra={"prediction_data": extra_record})

    return PredictionResponse(prediction=round(prediction_val, 2), unit="kg")
