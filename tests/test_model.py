"""Tests for model training artifacts and champion model inference (Stage 2)."""

from pathlib import Path
import joblib
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient


def test_champion_model_file_exists():
    """Verify that models/model.joblib exists on disk."""
    model_path = Path("models/model.joblib")
    assert model_path.exists(), "models/model.joblib must exist for deployment"


def test_champion_model_prediction():
    """Verify champion model loads and produces valid predictions on synthetic data."""
    model = joblib.load("models/model.joblib")
    sample = pd.DataFrame([{
        "worker_exp_yr": 12,
        "rain_mm_week": 145.0,
        "slope_deg": 18,
        "temp_c": 23.5,
        "bush_age_yr": 35,
        "hours_worked": 7.0,
        "flush": "first",
        "section": "River Side",
    }])
    preds = model.predict(sample)
    assert len(preds) == 1
    assert isinstance(float(preds[0]), float)
    assert 5.0 <= preds[0] <= 80.0


def test_mlflow_registry_alias():
    """Verify that the model is registered in sqlite:///mlflow.db with @champion alias."""
    client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
    model = client.get_registered_model("mp055-model")
    assert model is not None
    assert "champion" in model.aliases
    assert str(model.aliases["champion"]) == "1"
