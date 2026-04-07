from fastapi import FastAPI, HTTPException

import joblib
import numpy as np
import json

# Create the FastAPI app
app = FastAPI()

# Model variables to hold the loaded model and metadata (global variables)
model = None
model_metadata = None

# Load the model and metadata at api startup
@app.on_event("startup")
def load_model():
    # Load the pre-trained model 
    global model
    model = joblib.load('../models/best_model.pkl')

    # Load the model metadata (feature names) from a JSON file
    global model_metadata
    with open('../models/training_results.json', 'r') as f:
        model_metadata = json.load(f)

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
