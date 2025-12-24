import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime
import time
import random
import os


BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_openmeteo_data(start_date: str, end_date: str, latitude: float = 48.8534, longitude: float = 2.3488) -> pd.DataFrame:
    """
    Retrieve historical hourly weather data from Open-Meteo API in yearly chunks.
    
    The function:
    - Splits the requested date range into yearly chunks to avoid API limits.
    - Fetches hourly weather data for each chunk.
    - Converts the API response into a single pandas DataFrame.
    - Adds a datetime index and requested weather variables.
    
    Parameters:
    ----------
    start_date : str
        Start date of the data in "YYYY-MM-DD" format.
    end_date : str
        End date of the data in "YYYY-MM-DD" format.
    latitude : float, default 48.8534
        Latitude of the location (default is Paris coordinates).
    longitude : float, default 2.3488
        Longitude of the location (default is Paris coordinates).

    Returns:
    -------
    pd.DataFrame
        A DataFrame with the following columns:
        - 'datetime' (pd.Timestamp, UTC)
        - 'temperature_2m' (float, °C)
        - 'relative_humidity_2m' (float, %)
        - 'wind_speed_10m' (float, m/s)
        - 'shortwave_radiation_instant' (float, W/m²)
        Rows are indexed sequentially (0..n-1).

    Notes:
    -----
    - Uses requests-cache and retry_requests to handle temporary network errors.
    - Adds a small random delay between API calls to avoid rate limiting.
    - The datetime column is generated from the API's Unix timestamps and interval.
    - Suitable for multi-year data extraction.
    
    """
    # Setup Open-Meteo client with caching and retry
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    # Convert input dates to pandas datetime objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    all_dfs = []  # List to store DataFrames per year chunk

    # Loop over each year to avoid API request limits
    for year_start in pd.date_range(start, end, freq='YS'):
        # Define end of current year chunk, limited by overall end date
        year_end = min(year_start + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)

        # Define API request parameters
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": year_start.strftime("%Y-%m-%d"),
            "end_date": year_end.strftime("%Y-%m-%d"),
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "shortwave_radiation_instant"
            ]
        }

        print(f"Fetching: {params['start_date']} to {params['end_date']} ...")

        # Call Open-Meteo API
        responses = client.weather_api(BASE_URL, params=params)
        response = responses[0]
        hourly = response.Hourly()

        # Build DataFrame for the current chunk
        hourly_data = {
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

        df_chunk = pd.DataFrame(hourly_data)
        all_dfs.append(df_chunk)

        # Small random delay to avoid hitting API limits
        time.sleep(random.uniform(0.1, 0.5))

    # Concatenate all yearly chunks into a single DataFrame
    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df

# MAIN EXECUTION
if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("../data", exist_ok=True)
    df = get_openmeteo_data("2025-01-01", "2025-01-31")
    df.to_csv("../data/weather_jan2025.csv", index=False)
    print("Done!")