import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_BASE_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_BASE_PATH = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------
# Build full hourly datetime index
# ---------------------------------------------------------------------
def build_full_hourly_index(df: pd.DataFrame, time_col: str) -> pd.DatetimeIndex:
    return pd.date_range(
        start=df[time_col].min(),
        end=df[time_col].max(),
        freq="h"
    )

# ---------------------------------------------------------------------
# Reindex and interpolate time series data
# ---------------------------------------------------------------------
def reindex_and_interpolate_ts(
    df: pd.DataFrame,
    time_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:

    df = (
        df
        .drop_duplicates(subset=time_col)
        .sort_values(time_col)
        .copy()
    )

    df[time_col] = pd.to_datetime(df[time_col])

    full_index = build_full_hourly_index(df, time_col)

    df = (
        df
        .set_index(time_col)
        .reindex(full_index)
    )

    df[numeric_cols] = (
        df[numeric_cols]
        .interpolate(method="time", limit_area="inside")
    )

    if categorical_cols:
        df[categorical_cols] = (
            df[categorical_cols]
            .ffill()
            .bfill()
        )

    df = (
        df
        .rename_axis(time_col)
        .reset_index()
    )

    assert df[time_col].is_monotonic_increasing

    return df

# ---------------------------------------------------------------------
# Build processed dataset
# ---------------------------------------------------------------------
def build_processed_dataset_for_country_year(country: str, year: int):

    demand_path = (
        RAW_BASE_PATH
        / "electricity_demand"
        / f"country={country}"
        / f"year={year}"
        / "demand.parquet"
    )

    weather_path = (
        RAW_BASE_PATH
        / "weather"
        / f"country={country}"
        / f"year={year}"
        / "weather.parquet"
    )

    if not demand_path.exists() or not weather_path.exists():
        print(f"[SKIP] Missing raw data for {country} {year}")
        print(f"  Demand path : {demand_path}")
        print(f"  Weather path: {weather_path}")
        return

    print(f"[PROCESS] {country} {year}")

    df_demand = pd.read_parquet(demand_path)
    df_weather = pd.read_parquet(weather_path)

    # ---------------- Demand ----------------
    df_demand = reindex_and_interpolate_ts(
        df=df_demand,
        time_col="datetime",
        numeric_cols=["load_MW"],
        categorical_cols=["country"]
    )

    # ---------------- Weather ----------------
    weather_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation_instant"
    ]

    df_weather = reindex_and_interpolate_ts(
        df=df_weather,
        time_col="datetime",
        numeric_cols=weather_cols
    )

    # Weather: drop redundant metadata (single-country pipeline)
    df_weather = df_weather.drop(columns=["country"], errors="ignore")

    # ---------------- Merge ----------------
    df_processed = df_demand.merge(
        df_weather,
        on="datetime",
        how="inner"
    )

    df_processed["year"] = year

    # ---------------- Final checks ----------------
    assert df_processed.isna().sum().sum() == 0
    assert df_processed["country"].nunique() == 1

    # ---------------- Save ----------------
    output_dir = (
        PROCESSED_BASE_PATH
        / f"country={country}"
        / f"year={year}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "load_weather.parquet"
    df_processed.to_parquet(output_path, index=False)

    print(f"[SAVED] {output_path}")


def build_processed_dataset(countries: list[str], years: list[int]):
    for country in countries:
        for year in years:
            build_processed_dataset_for_country_year(country, year)

# ---------------------------------------------------------------------
# Command line execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    build_processed_dataset(
        countries=["FR"],
        years=[2023]
    )
