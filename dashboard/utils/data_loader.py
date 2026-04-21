"""
utils/data_loader.py - Centralised data loading functions for the dashboard.

All functions are cached with appropriate TTLs.
No data loading should happen outside this module.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REALTIME_PATH  = PROJECT_ROOT / "data" / "realtime" / "country=FR" / "realtime.parquet"
FEATURED_DIR   = PROJECT_ROOT / "data" / "featured" / "country=FR"
MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"
RESULTS_PATH   = PROJECT_ROOT / "models" / "training_results.json"


@st.cache_data(ttl=300)
def load_realtime() -> pd.DataFrame:
    """Load the realtime rolling window parquet (192 rows, 8-day window).

    Returns rows where load_MW is not null, sorted by datetime ascending.
    TTL: 5 minutes.
    """
    if not REALTIME_PATH.exists():
        raise FileNotFoundError(f"Realtime parquet not found: {REALTIME_PATH}")
    df = pd.read_parquet(REALTIME_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[df["load_MW"].notna()].reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_featured(year: int) -> pd.DataFrame:
    """Load the feature-engineered parquet for a given year.

    Args:
        year: Calendar year (e.g. 2024).

    Returns:
        DataFrame with all FEATURE_COLS + target_load_t+1, sorted by datetime.
    """
    path = FEATURED_DIR / f"year={year}" / "load_forecasting_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Featured parquet not found: {path}")
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_featured_range(year_start: int, year_end: int) -> pd.DataFrame:
    """Load and concatenate featured parquets for a range of years.

    Args:
        year_start: First year (inclusive).
        year_end:   Last year (inclusive).

    Returns:
        Concatenated DataFrame sorted by datetime.
    """
    frames = []
    for year in range(year_start, year_end + 1):
        try:
            frames.append(load_featured(year))
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError(
            f"No featured parquets found for years {year_start}–{year_end}"
        )
    return pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)


@st.cache_data(ttl=0)
def load_training_results() -> dict:
    """Load training_results.json (metrics, feature_cols, best_model name).

    TTL: 0 means cached for the lifetime of the session (file never changes
    after training).
    """
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"training_results.json not found: {RESULTS_PATH}")
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_latest_drift_report() -> dict | None:
    """Load the most recent drift monitoring JSON report.

    Returns None if no report exists yet.
    TTL: 5 minutes.
    """
    if not MONITORING_DIR.exists():
        return None
    reports = sorted(MONITORING_DIR.glob("*.json"))
    if not reports:
        return None
    with open(reports[-1], "r") as f:
        return json.load(f)