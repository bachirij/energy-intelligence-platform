import os
import pandas as pd


RAW_BASE_PATH = "data/raw"
PROCESSED_BASE_PATH = "data/processed"


def check_hourly_continuity(df: pd.DataFrame, time_col: str) -> pd.DatetimeIndex:
    """
    Build a complete hourly datetime index between min and max timestamps.
    """
    return pd.date_range(
        start=df[time_col].min(),
        end=df[time_col].max(),
        freq="h",
        tz="UTC"
    )


def interpolate_time_series(
    df: pd.DataFrame,
    value_cols: list[str],
    limit: int = 3
) -> pd.DataFrame:
    """
    Time-based linear interpolation with safety limit.
    """
    df[value_cols] = df[value_cols].interpolate(
        method="time",
        limit=limit,
        limit_direction="both"
    )
    return df


def build_processed_dataset_for_country_year(
    country: str,
    year: int
):
    """
    Build processed dataset (load + weather) for one country and one year.
    """

    demand_path = (
        f"{RAW_BASE_PATH}/electricity_demand/"
        f"country={country}/year={year}/demand.parquet"
    )

    weather_path = (
        f"{RAW_BASE_PATH}/weather/"
        f"country={country}/year={year}/weather.parquet"
    )

    if not os.path.exists(demand_path) or not os.path.exists(weather_path):
        print(f"[SKIP] Missing raw data for {country} {year}")
        return

    print(f"[PROCESS] {country} {year}")

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    df_demand = pd.read_parquet(demand_path)
    df_weather = pd.read_parquet(weather_path)

    # ------------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------------
    for df, name in [(df_demand, "demand"), (df_weather, "weather")]:
        if df["datetime"].dt.tz is None:
            raise ValueError(f"{name} datetime is not timezone-aware")

        if df["datetime"].dt.tz.zone != "UTC":
            raise ValueError(f"{name} datetime is not UTC")

        if df["datetime"].duplicated().any():
            raise ValueError(f"{name} contains duplicate timestamps")

    # ------------------------------------------------------------------
    # Reindex to full hourly timeline
    # ------------------------------------------------------------------
    full_index = check_hourly_continuity(df_demand, "datetime")

    df_demand = (
        df_demand
        .set_index("datetime")
        .reindex(full_index)
    )

    df_weather = (
        df_weather
        .set_index("datetime")
        .reindex(full_index)
    )

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    df_demand = interpolate_time_series(
        df_demand,
        value_cols=["load_MW"],
        limit=3
    )

    weather_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation_instant"
    ]

    df_weather = interpolate_time_series(
        df_weather,
        value_cols=weather_cols,
        limit=3
    )

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    df_processed = df_demand.join(df_weather, how="inner")

    # Restore columns
    df_processed = df_processed.reset_index().rename(
        columns={"index": "datetime"}
    )
    df_processed["country"] = country
    df_processed["year"] = year

    # ------------------------------------------------------------------
    # Final quality checks
    # ------------------------------------------------------------------
    missing_ratio = df_processed.isna().mean()

    if (missing_ratio > 0).any():
        print(
            f"[WARNING] Remaining NaNs for {country} {year}:\n"
            f"{missing_ratio[missing_ratio > 0]}"
        )

    # ------------------------------------------------------------------
    # Save processed data
    # ------------------------------------------------------------------
    output_dir = (
        f"{PROCESSED_BASE_PATH}/"
        f"country={country}/year={year}"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/load_weather.parquet"
    df_processed.to_parquet(output_path, index=False)

    print(f"[SAVED] {output_path}")


def build_processed_dataset(
    countries: list[str],
    years: list[int]
):
    """
    Build processed datasets for multiple countries and years.
    """
    for country in countries:
        for year in years:
            build_processed_dataset_for_country_year(country, year)


if __name__ == "__main__":
    build_processed_dataset(
        countries=["FR"],
        years=[2023]
    )
