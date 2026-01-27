"""
Open-Meteo Weather Ingestion
----------------------------

Fetch hourly historical weather data from the Open-Meteo Archive API
and store it as parquet files partitioned by country and year.

Output directory (always, regardless of where the script is run):
data/raw/weather/country=XX/year=YYYY/weather.parquet

Existing files are always overwritten (for the moment).
"""

from pathlib import Path
import time
import random
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests


# ---------------------------------------------------------------------
# Project paths 
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "weather"

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------
# Fetch one full year of weather data
# ---------------------------------------------------------------------
def fetch_openmeteo_weather_one_year(
    year: int,
    latitude: float,
    longitude: float
) -> pd.DataFrame:
    """
    Fetch hourly weather data for one full year from Open-Meteo.

    Parameters
    ----------
    year : int
        Year to fetch (e.g. 2023)
    latitude : float
        Latitude of the location
    longitude : float
        Longitude of the location

    Returns
    -------
    pd.DataFrame
        Columns:
        - datetime (UTC)
        - temperature_2m
        - relative_humidity_2m
        - wind_speed_10m
        - shortwave_radiation_instant
    """

    # Setup cached + retry-enabled session
    cache_session = requests_cache.CachedSession(
        cache_name=".cache/openmeteo",
        expire_after=-1
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "shortwave_radiation_instant"
        ]
    }

    responses = client.weather_api(BASE_URL, params=params)
    response = responses[0]
    hourly = response.Hourly()

    df = pd.DataFrame({
        "datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
        "shortwave_radiation_instant": hourly.Variables(3).ValuesAsNumpy()
    })

    if df.empty:
        raise ValueError("Open-Meteo returned an empty dataset")

    df = (
        df.sort_values("datetime")
          .drop_duplicates(subset=["datetime"])
          .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------------------
# Fetch and store 
# ---------------------------------------------------------------------
def fetch_openmeteo_weather_and_store(
    country: str,
    latitude: float,
    longitude: float,
    start_year: int,
    end_year: int
):
    """
    Fetch and store Open-Meteo weather data as parquet files
    partitioned by country and year.
    Files are always overwritten.
    """

    for year in range(start_year, end_year + 1):

        output_dir = (
            DATA_RAW_PATH
            / f"country={country}"
            / f"year={year}"
        )
        output_path = output_dir / "weather.parquet"

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[FETCH] Open-Meteo weather | {country} | {year}")

        try:
            df = fetch_openmeteo_weather_one_year(
                year=year,
                latitude=latitude,
                longitude=longitude
            )

            df["country"] = country

            df.to_parquet(output_path, index=False)

            print(
                f"[SAVED] {output_path} | rows={len(df)}"
            )

        except Exception as e:
            print(f"[ERROR] {country} {year} → {e}")

        time.sleep(random.uniform(0.2, 0.5))


# ---------------------------------------------------------------------
# Command line execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    fetch_openmeteo_weather_and_store(
        country="FR",
        latitude=48.8534,
        longitude=2.3488,
        start_year=2015,
        end_year=2024
    )
