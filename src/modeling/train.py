"""
Model Training — Electricity Load Forecasting (h+1)
----------------------------------------------------

Trains and compares multiple models on the feature-engineered dataset:
    - Ridge regression (linear baseline)
    - XGBoost
    - LightGBM

For each model:
    - Trains on 2015-2022
    - Evaluates on validation set (2023)
    - Reports MAE and RMSE

The best model is evaluated on the test set (2024) and saved to disk.

Output:
    models/best_model.pkl          <- best model pipeline
    models/training_results.json   <- metrics for all models

Usage:
    python src/modeling/train.py
    python main.py --steps train
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURED_BASE_PATH = PROJECT_ROOT / "data" / "featured"
MODELS_PATH = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
COUNTRY = "FR"
YEARS = list(range(2015, 2025))

FEATURE_COLS = [
    "load_t",
    "load_t-1",
    "load_t-24",
    "load_t-168",
    "temperature_t",
    "hour",
    "is_weekday",
    "week_of_year",
]
TARGET_COL = "target_load_t+1"

# Temporal splits
TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
VAL_END   = pd.Timestamp("2023-12-31 23:00:00", tz="UTC")


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_features(country: str, years: list) -> pd.DataFrame:
    """
    Load and concatenate feature-engineered parquet files.

    Returns a DataFrame indexed by datetime (UTC, sorted).
    """
    dfs = []
    for year in years:
        path = (
            FEATURED_BASE_PATH
            / f"country={country}"
            / f"year={year}"
            / "load_forecasting_features.parquet"
        )
        if path.exists():
            dfs.append(pd.read_parquet(path))
        else:
            print(f"[WARN] Missing features for {country} {year} — skipping")

    if not dfs:
        raise ValueError("No feature files found. Run the features step first.")

    df = (
        pd.concat(dfs, ignore_index=True)
          .sort_values("datetime")
          .set_index("datetime")
    )

    print(f"[DATA] Loaded {len(df):,} rows | {df.index.min()} → {df.index.max()}")
    return df


# ---------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    """
    Temporal split — never shuffle time series data.

    Train : 2015-2022
    Val   : 2023       (used for model selection)
    Test  : 2024       (used once, for final evaluation)
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train = X.loc[:TRAIN_END]
    y_train = y.loc[:TRAIN_END]

    X_val = X.loc[TRAIN_END:VAL_END]
    y_val = y.loc[TRAIN_END:VAL_END]

    X_test = X.loc[VAL_END:]
    y_test = y.loc[VAL_END:]

    print(f"[SPLIT] Train : {X_train.shape[0]:,} rows")
    print(f"[SPLIT] Val   : {X_val.shape[0]:,} rows")
    print(f"[SPLIT] Test  : {X_test.shape[0]:,} rows")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute MAE and RMSE."""
    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 2),
    }


# ---------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------
def build_models() -> dict:
    """
    Return a dict of named model pipelines to compare.

    Ridge uses a StandardScaler — mandatory for linear models.
    XGBoost and LightGBM are tree-based — no scaling needed,
    but we wrap them in a Pipeline for consistent API.
    """
    return {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=0.01)),
        ]),

        "xgboost": Pipeline([
            ("model", xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),

        "lightgbm": Pipeline([
            ("model", lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )),
        ]),
    }


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def train_and_compare(
    X_train, y_train,
    X_val, y_val,
) -> tuple[dict, str, object]:
    """
    Train all models and evaluate on the validation set.

    Returns
    -------
    results : dict
        Metrics for each model on the validation set
    best_name : str
        Name of the model with the lowest validation MAE
    best_model : fitted pipeline
        The best model, ready to be evaluated on test set
    """
    models = build_models()
    results = {}
    best_mae = float("inf")
    best_name = None
    best_model = None

    for name, pipeline in models.items():
        print(f"\n[TRAIN] {name} ...")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        metrics = evaluate(y_val, y_pred)
        results[name] = {"val": metrics}

        print(f"  VAL  MAE  = {metrics['mae']:>10.2f} MW")
        print(f"  VAL  RMSE = {metrics['rmse']:>10.2f} MW")

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_name = name
            best_model = pipeline

    return results, best_name, best_model


# ---------------------------------------------------------------------
# Final test evaluation + save
# ---------------------------------------------------------------------
def evaluate_and_save(
    best_model,
    best_name: str,
    X_test, y_test,
    results: dict,
) -> dict:
    """
    Evaluate the best model on the held-out test set (2024)
    and save the model + results to disk.
    """
    print(f"\n[TEST] Evaluating best model ({best_name}) on test set (2024) ...")

    y_test_pred = best_model.predict(X_test)
    test_metrics = evaluate(y_test, y_test_pred)
    results[best_name]["test"] = test_metrics

    print(f"  TEST MAE  = {test_metrics['mae']:>10.2f} MW")
    print(f"  TEST RMSE = {test_metrics['rmse']:>10.2f} MW")

    # Save model
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_PATH / "best_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n[SAVED] Model → {model_path}")

    # Save results
    results_payload = {
        "best_model": best_name,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "train_end": str(TRAIN_END),
        "val_end": str(VAL_END),
        "metrics": results,
    }

    results_path = MODELS_PATH / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[SAVED] Results → {results_path}")

    return results_payload


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def run_training(country: str = COUNTRY, years: list = YEARS):
    """
    Full training pipeline:
    1. Load features
    2. Split into train / val / test
    3. Train and compare Ridge, XGBoost, LightGBM
    4. Evaluate best model on test set
    5. Save model and metrics
    """
    print("=" * 60)
    print("  Training — Electricity Load Forecasting (h+1)")
    print("=" * 60)

    # Load data
    df = load_features(country=country, years=years)

    # Split
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    # Train all models and pick the best on val
    results, best_name, best_model = train_and_compare(
        X_train, y_train,
        X_val, y_val,
    )

    print(f"\n[BEST] Best model on validation: {best_name}")

    # Final evaluation on test + save
    results_payload = evaluate_and_save(
        best_model=best_model,
        best_name=best_name,
        X_test=X_test,
        y_test=y_test,
        results=results,
    )

    # Summary table
    print("\n" + "=" * 60)
    print("  Results summary")
    print("=" * 60)
    print(f"  {'Model':<12} {'Val MAE':>10} {'Val RMSE':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")
    for name, metrics in results_payload["metrics"].items():
        val = metrics.get("val", {})
        marker = " <-- best" if name == best_name else ""
        print(f"  {name:<12} {val.get('mae', '-'):>10} {val.get('rmse', '-'):>10}{marker}")
    print("=" * 60)

    return results_payload


# ---------------------------------------------------------------------
# Command line execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_training()