import os
import time
import random
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests


BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_openmeteo_weather_one_year(
    year: int,
    latitude: float,
    longitude: float
) -> pd.DataFrame:
    """
    Fetch hourly weather data for one year from Open-Meteo.
    """

    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
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

    data = {
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
    }

    return pd.DataFrame(data)


def fetch_openmeteo_weather_and_store(
    country: str,
    latitude: float,
    longitude: float,
    start_year: int,
    end_year: int,
    base_path: str = "data/raw/weather"
):
    """
    Fetch and store Open-Meteo weather data as parquet files
    partitioned by country and year.
    """

    for year in range(start_year, end_year + 1):

        output_dir = f"{base_path}/country={country}/year={year}"
        output_path = f"{output_dir}/weather.parquet"
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(output_path):
            print(f"[SKIP] {country} {year} already exists")
            continue

        print(f"[FETCH] Weather | {country} | {year}")

        try:
            df = fetch_openmeteo_weather_one_year(year, latitude, longitude)
            df["country"] = country
            df = df.sort_values("datetime")

            df.to_parquet(output_path, index=False)
            print(f"[SAVED] {output_path}")

        except Exception as e:
            print(f"[ERROR] {country} {year} → {e}")

        time.sleep(random.uniform(0.2, 0.5))


if __name__ == "__main__":
    fetch_openmeteo_weather_and_store(
        country="FR",
        latitude=48.8534,
        longitude=2.3488,
        start_year=2023,
        end_year=2024
    )
