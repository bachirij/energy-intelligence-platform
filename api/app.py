"""
FastAPI — Electricity Load Forecasting API
------------------------------------------

Exposes the trained XGBoost model as a REST API.

Endpoints:
    GET  /              -> API status
    GET  /health        -> model loading status
    GET  /model/info    -> model metadata (features, metrics)
    POST /predict       -> predict electricity load at h+1

Usage:
------
    uvicorn api.app:app --reload        # development
    uvicorn api.app:app --host 0.0.0.0  # production
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH   = PROJECT_ROOT / "models" / "best_model.pkl"
RESULTS_PATH = PROJECT_ROOT / "models" / "training_results.json"


# ---------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------
app = FastAPI(
    title="Energy Intelligence Platform",
    description="Hourly electricity load forecasting for France (h+1)",
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------
# The model is loaded once when the API starts, not at every request.
# This is critical for performance — loading a pickle at every call
# would make the API very slow.

model = None
model_metadata = None

@app.on_event("startup")
def load_model():
    """Load the trained model and its metadata at API startup."""
    global model, model_metadata

    if not MODEL_PATH.exists():
        print(f"[WARN] Model not found at {MODEL_PATH}. Call /predict will fail.")
        return

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r") as f:
            model_metadata = json.load(f)

    print(f"[OK] Model loaded from {MODEL_PATH}")


# ---------------------------------------------------------------------
# Request / Response schemas (Pydantic)
# ---------------------------------------------------------------------
# Pydantic models define what the API expects as input and returns
# as output. FastAPI uses them to validate requests automatically
# and generate the interactive documentation at /docs.

class PredictRequest(BaseModel):
    """
    Input features required to predict electricity load at h+1.

    All values correspond to the current hour t.
    """
    load_t: float = Field(..., description="Current electricity load in MW", example=45000.0)
    load_t_minus_1: float = Field(..., description="Load 1 hour ago in MW", example=44800.0)
    load_t_minus_24: float = Field(..., description="Load 24 hours ago in MW", example=43500.0)
    load_t_minus_168: float = Field(..., description="Load 168 hours ago (1 week) in MW", example=46200.0)
    temperature_t: float = Field(..., description="Current temperature in °C", example=12.5)
    hour: int = Field(..., ge=0, le=23, description="Current hour (0-23)", example=9)
    is_weekday: int = Field(..., ge=0, le=1, description="1 if weekday, 0 if weekend or holiday", example=1)
    week_of_year: int = Field(..., ge=1, le=52, description="Week of year (1-52)", example=12)


class PredictResponse(BaseModel):
    """API prediction response."""
    prediction_mw: float = Field(..., description="Predicted electricity load at h+1 in MW")
    prediction_datetime_utc: str = Field(..., description="Datetime for which the prediction is made (h+1)")
    model: str = Field(..., description="Model used for prediction")
    timestamp_utc: str = Field(..., description="When the prediction was made")


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    model_name: str | None


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.get("/", tags=["Status"])
def root():
    """
    Root endpoint — confirms the API is running.
    """
    return {
        "message": "Energy Intelligence Platform API is running.",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health():
    """
    Health check — confirms whether the model is loaded and ready.

    Returns status 'ok' if model is loaded, 'degraded' otherwise.
    """
    model_loaded = model is not None
    model_name = model_metadata.get("best_model") if model_metadata else None

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=model_name,
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    """
    Returns metadata about the model currently in production:
    feature names, training period, and validation/test metrics.
    """
    if model_metadata is None:
        raise HTTPException(
            status_code=404,
            detail="Model metadata not found. Make sure training_results.json exists.",
        )
    return model_metadata


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predict electricity load for the next hour (h+1).

    Accepts the current hour's features and returns a load forecast in MW.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run training first: python main.py --steps train",
        )

    # Build the feature vector in the exact same order as training
    # This is critical — a wrong column order silently produces wrong predictions
    features = pd.DataFrame([{
        "load_t":      request.load_t,
        "load_t-1":    request.load_t_minus_1,
        "load_t-24":   request.load_t_minus_24,
        "load_t-168":  request.load_t_minus_168,
        "temperature_t": request.temperature_t,
        "hour":        request.hour,
        "is_weekday":  request.is_weekday,
        "week_of_year": request.week_of_year,
    }])

    prediction = float(model.predict(features)[0])

    # h+1 datetime
    now_utc = datetime.now(timezone.utc)
    prediction_dt = now_utc.replace(minute=0, second=0, microsecond=0)
    prediction_dt = prediction_dt.replace(hour=(prediction_dt.hour + 1) % 24)

    model_name = model_metadata.get("best_model", "unknown") if model_metadata else "unknown"

    return PredictResponse(
        prediction_mw=round(prediction, 2),
        prediction_datetime_utc=prediction_dt.isoformat(),
        model=model_name,
        timestamp_utc=now_utc.isoformat(),
    )