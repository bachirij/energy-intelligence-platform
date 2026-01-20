"""
Feature Engineering for Electricity Load Forecasting (h+1)
----------------------------------------------------------

This module builds time-series features for short-term electricity load
forecasting (one hour ahead).

Main characteristics:
- No temporal leakage
- Multi-year continuous processing
- Lag-based, calendar-based and weather-based features
- Output partitioned by country and year

Output format:
    data/processed/country=XX/year=YYYY/load_forecasting_features.parquet
"""

from pathlib import Path
import pandas as pd
import holidays
from typing import List


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_BASE_PATH = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------
# Core feature engineering function
# ---------------------------------------------------------------------

def build_load_forecasting_features(
    country: str,
    years: List[int],
    forecast_horizon: int = 1,
) -> None:
    """
    Build feature datasets for short-term load forecasting.

    Parameters
    ----------
    country : str
        Country code (e.g. "FR")
    years : list[int]
        List of years to load (used to ensure continuity for lag features)
    forecast_horizon : int, default=1
        Forecast horizon in hours (currently designed for h+1)
    """

    # ------------------------------------------------------------
    # Load multi-year preprocessed data
    # ------------------------------------------------------------
    dfs = []

    for year in years:
        path = (
            PROCESSED_BASE_PATH
            / f"country={country}"
            / f"year={year}"
            / "load_weather.parquet"
        )

        if path.exists():
            dfs.append(pd.read_parquet(path))

    if not dfs:
        raise ValueError("No preprocessed data found for the given years")

    df = (
        pd.concat(dfs, ignore_index=True)
          .sort_values("datetime")
          .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Target variable (h+1)
    # ------------------------------------------------------------
    df[f"target_load_t+{forecast_horizon}"] = df["load_MW"].shift(-forecast_horizon)

    # ------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------
    df["hour"] = df["datetime"].dt.hour

    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekday"] = (df["day_of_week"] < 5).astype(int)

    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    # Replace ISO week 53 by 52 for consistency
    df.loc[df["week_of_year"] == 53, "week_of_year"] = 52

    # ------------------------------------------------------------
    # Holidays (treated as non-working days)
    # ------------------------------------------------------------
    country_holidays = holidays.country_holidays(country)

    df["is_holiday"] = df["datetime"].apply(
        lambda x: 1 if x.date() in country_holidays else 0
    )

    # Holidays override weekday flag
    df.loc[df["is_holiday"] == 1, "is_weekday"] = 0

    # ------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------
    df["load_t-1"] = df["load_MW"].shift(1)
    df["load_t-24"] = df["load_MW"].shift(24)
    df["load_t-168"] = df["load_MW"].shift(24 * 7)

    # ------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------
    # Rename the load column
    df = df.rename(columns = {"load_MW": "load_t", "temperature_2m": "temperature_t"})

    feature_cols = [
        "load_t",
        "load_t-1",
        "load_t-24",
        "load_t-168",
        "temperature_t",
        "hour",
        "is_weekday",
        "week_of_year",
    ]

    target_col = f"target_load_t+{forecast_horizon}"

    # Final dataset for modeling
    df_model = (
        df
        .assign(datetime=df["datetime"]) # Ensure datetime column is present as index
        .set_index("datetime")[feature_cols + [target_col]] # Select features and target
        .dropna() # Drop rows with missing values
        .copy()
    )

    # ------------------------------------------------------------
    # Final sanity checks
    # ------------------------------------------------------------
    if df_model.isna().sum().sum() != 0:
        raise ValueError("NaNs detected after feature engineering")

    if not df_model.index.is_monotonic_increasing:
        raise ValueError("Datetime index is not monotonic")
    
    if not df_model.index.is_unique:
        raise ValueError("Datetime index is not unique")

    # ------------------------------------------------------------
    # Save features partitioned by year
    # ------------------------------------------------------------
    for year, df_year in df_model.groupby(df_model.index.year):

        output_dir = (
            PROCESSED_BASE_PATH
            / f"country={country}"
            / f"year={year}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "load_forecasting_features.parquet"
        df_year.to_parquet(output_path, index=False)

        print(f"[SAVED] {output_path} | rows={len(df_year)}")


# ---------------------------------------------------------------------
# Command line execution
# ---------------------------------------------------------------------
if __name__ == "__main__":

    build_load_forecasting_features(
        country="FR",
        years=list(range(2012, 2026)),
        forecast_horizon=1,
    )
