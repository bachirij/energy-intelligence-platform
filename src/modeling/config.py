"""
Modeling Configuration — Electricity Load Forecasting (h+1)
------------------------------------------------------------

Single source of truth for:
- Feature columns and target
- Temporal splits (dynamic, based on current year)
- Years to load
- Model hyperparameters

All other modules import from here.
The FastAPI app also imports FEATURE_COLS from here to ensure consistency
between training and inference.
"""

from datetime import datetime
import pandas as pd


# ---------------------------------------------------------------------
# Country
# ---------------------------------------------------------------------
COUNTRY = "FR"


# ---------------------------------------------------------------------
# Dynamic temporal splits
# ---------------------------------------------------------------------
# Logic:
#   test  = most recent complete year (current_year - 1)
#   val   = year before test          (current_year - 2)
#   train = everything before val
#
# Example in 2026: test=2025, val=2024, train=2015-2023
# Example in 2027: test=2026, val=2025, train=2015-2024  ← no code change needed

_current_year = datetime.now().year
_test_year    = _current_year - 1
_val_year     = _current_year - 2

TRAIN_END = pd.Timestamp(f"{_val_year - 1}-12-31 23:00:00", tz="UTC")
VAL_END   = pd.Timestamp(f"{_val_year}-12-31 23:00:00",     tz="UTC")
TEST_END  = pd.Timestamp(f"{_test_year}-12-31 23:00:00",    tz="UTC")

YEARS = list(range(2015, _test_year + 1))  # 2015..2025 in 2026


# ---------------------------------------------------------------------
# Features and target
# ---------------------------------------------------------------------
# FEATURE_COLS is imported by:
#   - train.py    : to select columns from the feature dataset
#   - api/app.py  : to build the inference payload in the same order
#
# Any change here automatically propagates to both training and inference.

FEATURE_COLS = [
    "load_t",
    "load_t-1",
    "load_t-24",
    "load_t-168",
    "temperature_t",
    "temperature_t-24",
    "hour",
    "is_weekday",
    "day_of_week",
    "week_of_year",
]

TARGET_COL = "target_load_t+1"


# ---------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------
# Centralised here so they appear in training_results.json
# and can be tuned without touching models.py.

RIDGE_PARAMS = {
    "alpha": 0.01,
}

XGBOOST_PARAMS = {
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

LIGHTGBM_PARAMS = {
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}