"""
utils/prediction.py — Feature reconstruction and inference for the dashboard.

Replicates the /predict endpoint logic without the HTTP layer.
Reads directly from the realtime parquet via data_loader.

Note: this module intentionally duplicates the feature reconstruction logic
from api/main.py. A future refactor could extract this logic into
src/modeling/inference.py and share it between both consumers.
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_loader import load_realtime, load_training_results

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH   = PROJECT_ROOT / "models" / "best_model.pkl"


@st.cache_resource
def _load_model():
    """Load best_model.pkl once for the lifetime of the Streamlit server."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_next_hour() -> dict:
    """Reconstruct features from the realtime parquet and predict load at t+1.

    Returns a dict with:
        - predicted_load_MW (float)
        - target_datetime   (pd.Timestamp, UTC)
        - feature_values    (dict, for display/debug purposes)

    Raises:
        ValueError: if the realtime parquet does not contain enough rows
                    to reconstruct all lag features.
    """
    model        = _load_model()
    results      = load_training_results()
    feature_cols = results["feature_cols"]

    df = load_realtime()

    # The last row with a valid load is our reference point t
    last_row = df.iloc[-1]
    t        = last_row["datetime"]

    def _get_load_at(dt: pd.Timestamp) -> float:
        """Return load_MW at a specific datetime, raise if not found."""
        match = df[df["datetime"] == dt]
        if match.empty:
            raise ValueError(
                f"[PREDICT] No realtime row found for datetime {dt}. "
                "Realtime window may be too short to reconstruct all lag features."
            )
        return float(match.iloc[0]["load_MW"])

    # Reconstruct lag features by direct timestamp lookup
    load_t      = _get_load_at(t)
    load_t_1    = _get_load_at(t - pd.Timedelta(hours=1))
    load_t_24   = _get_load_at(t - pd.Timedelta(hours=24))
    load_t_168  = _get_load_at(t - pd.Timedelta(hours=168))

    # temperature_t is the forecast for t+1, stored at row t (shifted during ingestion)
    temperature_t = float(last_row["temperature_2m"])

    # temperature_t-24: forecast stored at row t-24
    row_t24 = df[df["datetime"] == (t - pd.Timedelta(hours=24))]
    if row_t24.empty or pd.isna(row_t24.iloc[0]["temperature_2m"]):
        # Fallback: use temperature_t as proxy (same forecast, close enough)
        temperature_t_24 = temperature_t
    else:
        temperature_t_24 = float(row_t24.iloc[0]["temperature_2m"])

    # Calendar features for target datetime t+1
    import holidays
    t_plus_1   = t + pd.Timedelta(hours=1)
    t_local    = t_plus_1.tz_convert("Europe/Paris")
    fr_holidays = holidays.country_holidays("FR")

    hour        = t_local.hour
    day_of_week = t_local.dayofweek
    week_of_year = t_local.isocalendar().week
    is_weekday  = 0 if (day_of_week >= 5 or t_local.date() in fr_holidays) else 1

    feature_values = {
        "load_t":           load_t,
        "load_t-1":         load_t_1,
        "load_t-24":        load_t_24,
        "load_t-168":       load_t_168,
        "temperature_t":    temperature_t,
        "temperature_t-24": temperature_t_24,
        "hour":             hour,
        "is_weekday":       is_weekday,
        "day_of_week":      day_of_week,
        "week_of_year":     int(week_of_year),
    }

    # Build input row respecting the exact feature order from training
    X = pd.DataFrame([[feature_values[col] for col in feature_cols]], columns=feature_cols)

    predicted_load = float(model.predict(X)[0])

    return {
        "predicted_load_MW": predicted_load,
        "target_datetime":   t_plus_1,
        "feature_values":    feature_values,
    }