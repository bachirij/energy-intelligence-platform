import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def make_realtime_df(n_hours: int = 200) -> pd.DataFrame:
    """
    Minimal realtime parquet content with enough rows to cover t-168h.
    """
    base = pd.Timestamp("2026-01-05 10:00:00", tz="UTC")  # Monday
    datetimes = [base + pd.Timedelta(hours=i) for i in range(n_hours)]
    return pd.DataFrame({
        "datetime": datetimes,
        "load_MW": [50000.0 + i * 10 for i in range(n_hours)],
        "temperature_2m": [5.0 + i * 0.01 for i in range(n_hours)],
    })


def make_mock_model() -> MagicMock:
    """Fake sklearn model that always predicts 45000.0 MW."""
    mock = MagicMock()
    mock.predict.return_value = np.array([45000.0])
    return mock


def make_mock_metadata() -> dict:
    return {
        "best_model": "xgboost",
        "feature_cols": [
            "load_t", "load_t-1", "load_t-24", "load_t-168",
            "temperature_t", "temperature_t-24",
            "hour", "is_weekday", "day_of_week", "week_of_year",
        ],
    }


# -----------------------------------------------------------------------
# Tests — /health
# -----------------------------------------------------------------------
def test_health_returns_200():
    """
    /health must return 200 with status, best_model and feature_cols
    when the model is loaded.
    """
    with patch("api.main.model", make_mock_model()), \
         patch("api.main.model_metadata", make_mock_metadata()):

        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["best_model"] == "xgboost"
    assert "feature_cols" in body


def test_health_returns_500_if_model_not_loaded():
    """
    /health must return 500 if model is None.
    """
    with patch("api.main.model", None):
        response = client.get("/health")

    assert response.status_code == 500


# -----------------------------------------------------------------------
# Tests — /predict
# -----------------------------------------------------------------------
def test_predict_returns_200_with_correct_fields():
    """
    /predict must return 200 with predicted_load_MW, predicted_at
    and target_datetime when all data is available.
    """
    df = make_realtime_df(n_hours=200)

    with patch("api.main.model", make_mock_model()), \
         patch("api.main.model_metadata", make_mock_metadata()), \
         patch("api.main.fr_holidays", set()), \
         patch("api.main.BASE_DIR") as mock_base_dir, \
         patch("pandas.read_parquet", return_value=df):

        mock_base_dir.__truediv__ = lambda self, other: MagicMock(exists=lambda: True)

        response = client.get("/predict")

    assert response.status_code == 200
    body = response.json()
    assert "predicted_load_MW" in body
    assert "predicted_at" in body
    assert "target_datetime" in body
    assert body["predicted_load_MW"] == 45000.0


def test_predict_returns_503_if_parquet_missing():
    """
    /predict must return 503 if the realtime parquet does not exist.
    """
    with patch("api.main.model", make_mock_model()), \
         patch("api.main.model_metadata", make_mock_metadata()), \
         patch("api.main.fr_holidays", set()):

        # Patch the path to a non-existent file
        with patch("pathlib.Path.exists", return_value=False):
            response = client.get("/predict")

    assert response.status_code == 503