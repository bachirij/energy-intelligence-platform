"""
Real-Time Data Ingestion
------------------------

Fetches the latest electricity demand (ENTSO-E) and weather forecast
(Open-Meteo) needed to build features and run an h+1 prediction.

What this script fetches:
- ENTSO-E  : actual demand for the last 48 hours (lag features need up to t-168h,
             but this script is designed to be run every hour on top of existing
             historical data — 48h is enough to cover any short gap or retry)
- Open-Meteo: hourly weather forecast for the next 2 hours (we need t and t+1)

Output:
    data/realtime/country=XX/realtime.parquet
    (a rolling window — older rows are dropped to keep the file lightweight)

Usage:
    python src/ingestion/get_realtime_data.py
    python main.py --steps realtime
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import requests_cache
import openmeteo_requests
from retry_requests import retry
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REALTIME_PATH = PROJECT_ROOT / "data" / "realtime"

ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Rolling window: how many hours of realtime data to keep on disk
ROLLING_WINDOW_HOURS = 24 * 7 + 24 # 7 + 1 days


# ---------------------------------------------------------------------
# ENTSO-E: fetch demand for the last N hours
# ---------------------------------------------------------------------
def fetch_entsoe_realtime(
    country_code: str,
    api_token: str,
    lookback_hours: int = 168 + 24,  # 7 days + 1 day buffer
) -> pd.DataFrame:
    """
    Fetch actual electricity demand from ENTSO-E for the last N hours.

    Parameters
    ----------
    country_code : str
        ENTSO-E bidding zone code (e.g. "10YFR-RTE------C")
    api_token : str
        ENTSO-E API token
    lookback_hours : int
        How many hours back to fetch (default: 48)

    Returns
    -------
    pd.DataFrame with columns: datetime (UTC, tz-aware), load_MW
    """
    now_utc = datetime.now(timezone.utc)
    start = now_utc - timedelta(hours=lookback_hours)

    # ENTSO-E expects format: YYYYMMDDHHmm
    period_start = start.strftime("%Y%m%d%H%M")
    period_end = now_utc.strftime("%Y%m%d%H%M")

    print(f"[ENTSOE] Fetching demand from {period_start} to {period_end} UTC")

    params = {
        "documentType": "A65",                    # Actual Total Load
        "processType": "A16",                     # Realised
        "outBiddingZone_Domain": country_code,
        "periodStart": period_start,
        "periodEnd": period_end,
        "securityToken": api_token,
    }

    response = requests.get(ENTSOE_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    ns = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}

    period_nodes = root.findall(".//ns:Period", ns)
    if not period_nodes:
        raise ValueError(
            "No <Period> nodes found in ENTSO-E response. "
            "The data may not be published yet for this time window."
        )

    records = []
    for period in period_nodes:
        start_str = period.find("ns:timeInterval/ns:start", ns).text
        start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))

        for point in period.findall("ns:Point", ns):
            position = int(point.find("ns:position", ns).text)
            quantity = float(point.find("ns:quantity", ns).text)
            timestamp = start_time + pd.Timedelta(hours=position - 1)
            records.append({"datetime": timestamp, "load_MW": quantity})

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("ENTSO-E returned an empty dataset for the requested window.")

    # Filter to requested window (in case ENTSO-E returns extra data beyond period_end)
    df = df[(df["datetime"] >= start) & (df["datetime"] <= now_utc)].copy()

    df = (
        df.sort_values("datetime")
          .drop_duplicates(subset=["datetime"])
          .reset_index(drop=True)
    )

    # Resample to strict hourly frequency
    # Handles sub-hourly points introduced by ENTSO-E during DST transitions
    print(f"[DEBUG] Before resample : {len(df)} lines")
    df = (
        df.set_index("datetime")
          .resample("1h")["load_MW"]
          .mean()
          .reset_index()
    )
    print(f"[DEBUG] After resample : {len(df)} lines")

    print(f"[ENTSOE] Fetched {len(df)} hourly rows")
    return df


# ---------------------------------------------------------------------
# Open-Meteo: fetch weather forecast for the next few hours
# ---------------------------------------------------------------------
def fetch_openmeteo_realtime(
    latitude: float,
    longitude: float,
    forecast_hours: int = 2,
) -> pd.DataFrame:
    """
    Fetch hourly weather forecast from Open-Meteo for the next N hours.

    Uses the /forecast endpoint (not /archive), which provides:
    - Current conditions (t)
    - Short-term forecast (t+1, t+2, ...)

    Parameters
    ----------
    latitude : float
    longitude : float
    forecast_hours : int
        How many forecast hours to fetch (default: 2, covers t and t+1)

    Returns
    -------
    pd.DataFrame with columns:
        datetime (UTC, tz-aware), temperature_2m, relative_humidity_2m,
        wind_speed_10m, shortwave_radiation_instant
    """
    print(f"[OPENMETEO] Fetching weather forecast for lat={latitude}, lon={longitude}")

    cache_session = requests_cache.CachedSession(
        cache_name=".cache/openmeteo_realtime",
        expire_after=1800  # cache expires after 30 minutes
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "shortwave_radiation_instant",
        ],
        "forecast_hours": forecast_hours,  # only fetch what we need
        "timezone": "UTC",
    }

    responses = client.weather_api(OPENMETEO_FORECAST_URL, params=params)
    response = responses[0]
    hourly = response.Hourly()

    df = pd.DataFrame({
        "datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
        "shortwave_radiation_instant": hourly.Variables(3).ValuesAsNumpy(),
    })

    if df.empty:
        raise ValueError("Open-Meteo returned an empty forecast.")

    df = (
        df.sort_values("datetime")
          .drop_duplicates(subset=["datetime"])
          .reset_index(drop=True)
    )

    print(f"[OPENMETEO] Fetched {len(df)} forecast rows")
    return df


# ---------------------------------------------------------------------
# Merge demand + weather into a single realtime snapshot
# ---------------------------------------------------------------------
def build_realtime_snapshot(
    df_demand: pd.DataFrame,
    df_weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge demand and weather on datetime and return a clean snapshot.

    Both DataFrames must have a tz-aware UTC datetime column.
    The merge is a left join on demand, so we only keep rows where
    demand data exists (weather forecast is forward-looking).

    Returns
    -------
    pd.DataFrame — merged snapshot, sorted by datetime
    """
    # Normalize both datetime columns to UTC hour precision
    df_demand = df_demand.copy()
    df_weather = df_weather.copy()

    df_demand["datetime"] = pd.to_datetime(df_demand["datetime"], utc=True).dt.floor("h")
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"], utc=True).dt.floor("h")

    # Substract 1 hour from weather timestamps to align with demand (weather at t is used to predict demand at t+1)
    df_weather['datetime'] = df_weather['datetime'] - pd.Timedelta(hours=1)

    df_merged = df_demand.merge(df_weather, on="datetime", how="left")

    df_merged = (
        df_merged.sort_values("datetime")
                 .drop_duplicates(subset=["datetime"])
                 .reset_index(drop=True)
    )

    return df_merged


# ---------------------------------------------------------------------
# Save with rolling window logic
# ---------------------------------------------------------------------
def save_realtime_snapshot(
    df_new: pd.DataFrame,
    country: str,
    rolling_window_hours: int = ROLLING_WINDOW_HOURS,
) -> None:
    """
    Append new rows to the realtime parquet file and drop rows older
    than the rolling window.

    Parameters
    ----------
    df_new : pd.DataFrame
        New snapshot to append
    country : str
        Country code (used for partitioning)
    rolling_window_hours : int
        Maximum age of rows to keep (default: 7 days)
    """
    output_dir = REALTIME_PATH / f"country={country}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "realtime.parquet"

    df_new["country"] = country

    if output_path.exists():
        df_existing = pd.read_parquet(output_path)

        # Ensure consistent datetime format before concat
        df_existing["datetime"] = pd.to_datetime(df_existing["datetime"], utc=True)
        df_new["datetime"] = pd.to_datetime(df_new["datetime"], utc=True)

        df_combined = (
            pd.concat([df_existing, df_new], ignore_index=True)
              .drop_duplicates(subset=["datetime"])
              .sort_values("datetime")
              .reset_index(drop=True)
        )
    else:
        df_new["datetime"] = pd.to_datetime(df_new["datetime"], utc=True)
        df_combined = df_new

    # Apply rolling window: drop rows older than N hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=rolling_window_hours)
    df_combined = df_combined[df_combined["datetime"] >= cutoff].reset_index(drop=True)

    df_combined.to_parquet(output_path, index=False)

    print(f"[SAVED] {output_path} | total rows in window = {len(df_combined)}")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def fetch_and_store_realtime(
    country: str,
    country_code: str,
    latitude: float,
    longitude: float,
) -> None:
    """
    Full real-time ingestion pipeline:
    1. Fetch last 48h of demand from ENTSO-E
    2. Fetch next 2h weather forecast from Open-Meteo
    3. Merge both into a snapshot
    4. Append to rolling parquet file

    Parameters
    ----------
    country : str
        Country code label (e.g. "FR")
    country_code : str
        ENTSO-E bidding zone code (e.g. "10YFR-RTE------C")
    latitude : float
    longitude : float
    """
    load_dotenv()
    api_token = os.getenv("ENTSOE_API_TOKEN")
    if not api_token:
        raise ValueError("ENTSOE_API_TOKEN not found in environment variables.")

    print(f"\n[REALTIME] Starting real-time ingestion | country={country}")
    print(f"[REALTIME] Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # Step 1 — ENTSO-E demand (last 168 + 24h)
    df_demand = fetch_entsoe_realtime(
        country_code=country_code,
        api_token=api_token,
        lookback_hours=168 + 24,  # fetch 7 days + 1 day buffer to ensure we have all lag features covered
    )

    # Step 2 — Open-Meteo forecast (next 2h)
    df_weather = fetch_openmeteo_realtime(
        latitude=latitude,
        longitude=longitude,
        forecast_hours=2,
    )

    # Step 3 — Merge
    df_snapshot = build_realtime_snapshot(df_demand, df_weather)
    print(f"[REALTIME] Snapshot built | rows={len(df_snapshot)}")

    # Step 4 — Save with rolling window
    save_realtime_snapshot(df_snapshot, country=country)

    print(f"[REALTIME] Done.")


# ---------------------------------------------------------------------
# Command line execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    fetch_and_store_realtime(
        country="FR",
        country_code="10YFR-RTE------C",
        latitude=48.8534,
        longitude=2.3488,
    )