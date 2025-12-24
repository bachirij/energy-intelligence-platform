import pandas as pd
import holidays


# 1. Cleaning functions
def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess weather data."""
    df = df.copy()

    # Ensure datetime is in proper format
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # Interpolate missing values if needed
    df = df.set_index("datetime").interpolate(method="time").reset_index()

    # Ensure uniform hourly frequency
    full_index = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df = df.set_index("datetime").reindex(full_index).interpolate(method="time").reset_index()
    df = df.rename(columns={"index": "datetime"})

    # Sanity checks
    assert df["datetime"].isna().sum() == 0, "NaT values found in datetime"
    assert (df["datetime"].diff().dropna() == pd.Timedelta(hours=1)).all(), "Non-hourly intervals detected"

    return df


def clean_load_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess electricity load data."""
    df = df.copy()

    # Ensure datetime is in proper format
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # Drop duplicates (sometimes API returns overlapping data)
    df = df.drop_duplicates(subset=["datetime"])

    # Reindex to ensure continuous hourly timestamps
    full_index = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df = df.set_index("datetime").reindex(full_index)

    # Interpolate missing load values
    df["load_MW"] = df["load_MW"].interpolate(method="time")

    df = df.reset_index().rename(columns={"index": "datetime"})

    # Sanity checks
    assert df["datetime"].isna().sum() == 0, "NaT values found in datetime"
    assert (df["datetime"].diff().dropna() == pd.Timedelta(hours=1)).all(), "Non-hourly intervals detected"

    return df


# 2. Merge function
def merge_weather_load(df_weather: pd.DataFrame, df_load: pd.DataFrame) -> pd.DataFrame:
    """Merge weather and load data on datetime."""
    df_merged = pd.merge(df_load, df_weather, on="datetime", how="inner")

    # Sanity checks
    assert df_merged["datetime"].isna().sum() == 0, "NaN datetime after merge"
    assert (df_merged["datetime"].diff().dropna() == pd.Timedelta(hours=1)).all(), "Non-hourly frequency after merge"

    return df_merged


# 3. Feature engineering functions
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal, calendar, and categorical features."""
    df = df.copy()
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)

    # Holidays and weekends
    fr_holidays = holidays.FR()
    df["is_holiday"] = df["datetime"].dt.date.astype("datetime64[ns]").isin(fr_holidays).astype(int)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Season
    def get_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    df["season"] = df["month"].apply(get_season)

    return df


# 4. Full preprocessing function
def preprocess_all(df_weather_raw: pd.DataFrame, df_load_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Cleans weather and load data
    - Merges both datasets
    - Adds engineered features
    """
    df_weather = clean_weather_data(df_weather_raw)
    df_load = clean_load_data(df_load_raw)
    df_merged = merge_weather_load(df_weather, df_load)
    df_final = add_time_features(df_merged)

    return df_final


# 5. MAIN EXECUTION
if __name__ == "__main__":
    from src.ingestion.import_meteo import get_openmeteo_data
    from src.ingestion.import_entsoe import get_entsoe_load
    import os

    START_DATE = "2025-01-01"
    END_DATE = "2025-01-02"
    os.makedirs("../data", exist_ok=True)

    print("Fetching raw data...")
    df_weather_raw = get_openmeteo_data(START_DATE, END_DATE)
    df_load_raw = get_entsoe_load(START_DATE, END_DATE)

    print("Preprocessing data...")
    df_inputs = preprocess_all(df_weather_raw, df_load_raw)

    df_inputs.to_csv("../data/preprocessed_inputs.csv", index=False)
    print("Preprocessed data saved to '../data/preprocessed_inputs.csv'")

