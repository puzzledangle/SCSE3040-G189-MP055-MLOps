"""Pydantic schemas for the MP055 prediction API.

Enforces the client's business rules:
1. bush_age_yr <= 150 (impossible beyond 150 years; reject with 422)
2. worker_exp_yr >= 0 (negative experience is always a typing error; reject with 422)
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Input payload for POST /predict."""

    model_config = ConfigDict(extra="forbid")

    worker_exp_yr: float = Field(
        ...,
        ge=0,
        description="Worker's experience in years (must be >= 0)",
    )
    rain_mm_week: float = Field(
        ...,
        description="Rainfall in the past week in mm",
    )
    slope_deg: float = Field(
        ...,
        description="Slope of the tea section in degrees",
    )
    temp_c: float = Field(
        ...,
        description="Average daily temperature in degrees Celsius",
    )
    bush_age_yr: float = Field(
        ...,
        le=150,
        description="Age of tea bushes in years (must be <= 150)",
    )
    hours_worked: float = Field(
        ...,
        gt=0,
        description="Hours plucked during the shift (must be > 0)",
    )
    flush: Literal["first", "second", "monsoon", "first ", " second", "mon.", "fir.", "sec.", "FIRST", "SECOND", "MONSOON", "First", "Second", "Monsoon"] = Field(
        ...,
        description="Picking season flush category",
    )
    section: Literal["Upper Garden", "Lower Garden", "River Side"] = Field(
        ...,
        description="Operational estate section",
    )


class PredictionResponse(BaseModel):
    """Output payload for POST /predict."""

    prediction: float = Field(
        ...,
        description="Predicted leaf yield plucked in kg (kg_per_worker)",
    )
    unit: str = Field(
        default="kg",
        description="Unit of measurement",
    )


class HealthResponse(BaseModel):
    """Output payload for GET /health."""

    status: str = Field(default="ok", description="Service health status")
    model: str = Field(default="mp055-model", description="Registered model name")
    version: str = Field(default="champion", description="Active model alias")
