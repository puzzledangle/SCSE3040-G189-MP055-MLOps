"""Unit tests for FastAPI prediction service using TestClient (Stage 3)."""

from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_predict_valid_request():
    """Verify POST /predict returns 200 with realistic yield prediction."""
    payload = {
        "worker_exp_yr": 8.0,
        "rain_mm_week": 135.0,
        "slope_deg": 16.0,
        "temp_c": 23.0,
        "bush_age_yr": 30.0,
        "hours_worked": 7.5,
        "flush": "first",
        "section": "Upper Garden",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert 5.0 <= data["prediction"] <= 70.0


def test_rule_1_bush_age_above_150_rejected():
    """Client rule 1: bush_age_yr > 150 must be rejected with 422."""
    payload = {
        "worker_exp_yr": 8.0,
        "rain_mm_week": 135.0,
        "slope_deg": 16.0,
        "temp_c": 23.0,
        "bush_age_yr": 151.0,  # Invalid: > 150
        "hours_worked": 7.5,
        "flush": "first",
        "section": "Upper Garden",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_rule_2_worker_exp_below_0_rejected():
    """Client rule 2: worker_exp_yr < 0 must be rejected with 422."""
    payload = {
        "worker_exp_yr": -1.0,  # Invalid: < 0
        "rain_mm_week": 135.0,
        "slope_deg": 16.0,
        "temp_c": 23.0,
        "bush_age_yr": 30.0,
        "hours_worked": 7.5,
        "flush": "first",
        "section": "Upper Garden",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_required_field():
    """Verify that requests missing required fields are rejected with 422."""
    incomplete_payload = {
        "worker_exp_yr": 8.0,
        "rain_mm_week": 135.0,
        # missing slope_deg, temp_c, etc.
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_flush_variant_handling():
    """Verify that flush variants like 'fir.' or 'Monsoon' are handled gracefully."""
    payload = {
        "worker_exp_yr": 5.0,
        "rain_mm_week": 160.0,
        "slope_deg": 12.0,
        "temp_c": 26.0,
        "bush_age_yr": 20.0,
        "hours_worked": 8.0,
        "flush": "fir.",  # Variant
        "section": "River Side",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
