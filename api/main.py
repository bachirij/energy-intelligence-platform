import joblib
import json
import numpy as np
import pandas as pd
import holidays
from datetime import timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException

# Create the FastAPI app
app = FastAPI()

# Model variables to hold the loaded model and metadata (global variables)
model = None
model_metadata = None
fr_holidays = None

# Load the model and metadata at api startup
@app.on_event("startup")
def load_model():
    global model, model_metadata, fr_holidays
    # Load the pre-trained model 
    model = joblib.load('../models/best_model.pkl')

    # Load the model metadata (feature names) from a JSON file
    with open('../models/training_results.json', 'r') as f:
        model_metadata = json.load(f)

    # Load France holidays for calendar features
    fr_holidays = holidays.country_holidays("FR")

# GET /health endpoint to check if the API is running
@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if model_metadata is None:
        raise HTTPException(status_code=500, detail="Model metadata not loaded")

    return {"status": "ok", 
            "best_model": model_metadata.get("best_model"),
            "feature_cols": model_metadata.get("feature_cols", [])}

# GET /predict endpoint to make predictions based on query parameters
@app.get("/predict")
async def predict():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # ------------------------------------------------------------------
    # 1. Load realtime parquet
    # ------------------------------------------------------------------
    realtime_path = Path("../data/realtime/country=FR/realtime.parquet")
    if not realtime_path.exists():
        raise HTTPException(status_code=503, detail="Realtime data not available")

    df = pd.read_parquet(realtime_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Identify t (last row)
    # ------------------------------------------------------------------
    t = df["datetime"].iloc[-1]

    def lookup(col: str, delta_hours: int) -> float:
        """Lookup a column value at timestamp t - delta_hours."""
        target_ts = t - pd.Timedelta(hours=delta_hours)
        row = df.loc[df["datetime"] == target_ts, col]
        if row.empty:
            raise HTTPException(
                status_code=503,
                detail=f"Missing data for {col} at t-{delta_hours}h ({target_ts})"
            )
        return float(row.iloc[0])

    # ------------------------------------------------------------------
    # 3. Build features (same order as FEATURE_COLS in src/modeling/config.py)
    # ------------------------------------------------------------------
    # Convert t to Europe/Paris for calendar features
    # (holidays/weekday logic uses local time, consistent with training)
    t_paris = t.astimezone(timezone.utc).replace(tzinfo=None)
    t_paris = pd.Timestamp(t_paris).tz_localize("UTC").tz_convert("Europe/Paris")

    try:
        features = {
            "load_t":           lookup("load_MW", 0),
            "load_t-1":         lookup("load_MW", 1),
            "load_t-24":        lookup("load_MW", 24),
            "load_t-168":       lookup("load_MW", 168),
            "temperature_t":    lookup("temperature_2m", 0),
            "temperature_t-24": lookup("temperature_2m", 24),
            "hour":             t_paris.hour,
            "is_weekday":       int(t_paris.dayofweek < 5 and t_paris.date() not in fr_holidays),
            "day_of_week":      t_paris.dayofweek,
            "week_of_year":     t_paris.isocalendar().week,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature construction failed: {e}")

    # ------------------------------------------------------------------
    # 4. Assemble in model's expected column order
    # ------------------------------------------------------------------
    feature_cols = model_metadata.get("feature_cols")
    if not feature_cols:
        raise HTTPException(status_code=500, detail="feature_cols missing from metadata")

    try:
        X = pd.DataFrame([features])[feature_cols]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Feature mismatch: {e}")

    # ------------------------------------------------------------------
    # 5. Predict
    # ------------------------------------------------------------------
    prediction_mw = float(model.predict(X)[0])
    t_plus_1 = t + pd.Timedelta(hours=1)

    return {
        "predicted_at": t.isoformat(),
        "target_datetime": t_plus_1.isoformat(),
        "predicted_load_MW": round(prediction_mw, 1),
    }
