"""
Storage — Electricity Load Forecasting (h+1)
--------------------------------------------

All disk I/O in one place:
- Reading feature parquet files
- Saving the trained model (.pkl)
- Saving training results (.json)

If the storage layer changes (e.g. moving to a database or cloud storage),
only this file needs to be updated.
"""

import json
import pickle
from pathlib import Path

import pandas as pd

from src.modeling.config import COUNTRY, FEATURE_COLS, TARGET_COL, TRAIN_END, VAL_END


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
FEATURED_BASE   = PROJECT_ROOT / "data" / "featured"
MODELS_PATH     = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_features(country: str, years: list[int]) -> pd.DataFrame:
    """
    Load and concatenate feature-engineered parquet files for all requested years.

    Parameters
    ----------
    country : str
        Country code (e.g. "FR").
    years : list[int]
        Years to load (e.g. [2015, ..., 2025]).

    Returns
    -------
    pd.DataFrame
        Sorted by datetime, indexed by datetime (UTC-aware).
    """
    dfs = []

    for year in years:
        path = (
            FEATURED_BASE
            / f"country={country}"
            / f"year={year}"
            / "load_forecasting_features.parquet"
        )
        if path.exists():
            dfs.append(pd.read_parquet(path))
        else:
            print(f"[WARN] Missing features for {country} {year} — skipping")

    if not dfs:
        raise ValueError(
            f"No feature files found for {country} {years}. "
            "Run the feature engineering step first."
        )

    df = (
        pd.concat(dfs, ignore_index=True)
          .sort_values("datetime")
          .set_index("datetime")
    )

    print(f"[DATA] Loaded {len(df):,} rows | {df.index.min()} → {df.index.max()}")
    return df


# ---------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------

def split_data(df: pd.DataFrame) -> tuple:
    """
    Temporal train / val / test split — never shuffle time series data.

    Boundaries are defined in config.py and update automatically each year:
        train : 2015 → TRAIN_END   (val_year - 1, Dec 31 23:00 UTC)
        val   : TRAIN_END+1h → VAL_END
        test  : VAL_END+1h  → end of data

    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataset indexed by datetime.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    one_hour = pd.Timedelta("1h")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train = X.loc[:TRAIN_END]
    y_train = y.loc[:TRAIN_END]

    X_val   = X.loc[TRAIN_END + one_hour : VAL_END]
    y_val   = y.loc[TRAIN_END + one_hour : VAL_END]

    X_test  = X.loc[VAL_END + one_hour :]
    y_test  = y.loc[VAL_END + one_hour :]

    print(f"[SPLIT] Train : {len(X_train):,} rows  ({X_train.index.min().year}–{X_train.index.max().year})")
    print(f"[SPLIT] Val   : {len(X_val):,} rows  ({X_val.index.min().year})")
    print(f"[SPLIT] Test  : {len(X_test):,} rows  ({X_test.index.min().year})")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------
# Model and results persistence
# ---------------------------------------------------------------------

def save_model_and_results(
    model,
    model_name: str,
    results: dict,
) -> None:
    """
    Save the best model as a .pkl and all metrics as a .json.

    Parameters
    ----------
    model : fitted sklearn Pipeline
        The best model selected on the validation set.
    model_name : str
        Name of the best model (used in the JSON).
    results : dict
        Full metrics dict {model_name: {"val": {...}, "test": {...}}}.
    """
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = MODELS_PATH / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[SAVED] Model   → {model_path}")

    # Results + metadata
    payload = {
        "best_model":   model_name,
        "feature_cols": FEATURE_COLS,
        "target_col":   TARGET_COL,
        "train_end":    str(TRAIN_END),
        "val_end":      str(VAL_END),
        "metrics":      results,
    }

    results_path = MODELS_PATH / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVED] Results → {results_path}")