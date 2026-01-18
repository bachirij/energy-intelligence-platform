"""
ENTSO-E Electricity Demand Ingestion
------------------------------------

Fetch hourly actual electricity demand from the ENTSO-E Transparency Platform
and store it as parquet files partitioned by country and year.

Output directory (always, regardless of where the script is run):
data/raw/electricity_demand/country=XX/year=YYYY/demand.parquet

Existing files are always overwritten (for the moment).
"""

from pathlib import Path
import time
import random
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from dotenv import load_dotenv
import os


# ---------------------------------------------------------------------
# Project paths 
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "electricity_demand"

BASE_URL = "https://web-api.tp.entsoe.eu/api"


# ---------------------------------------------------------------------
# Fetch one full year of demand
# ---------------------------------------------------------------------
def fetch_entsoe_demand_one_year(
    year: int,
    country_code: str,
    api_token: str
) -> pd.DataFrame:
    """
    Fetch hourly actual electricity demand for one full year from ENTSO-E.

    Returns
    -------
    pd.DataFrame with columns:
    - datetime (UTC)
    - load_MW
    """

    period_start = f"{year}01010000"
    period_end = f"{year + 1}01010000"

    params = {
        "documentType": "A65",   # Actual Total Load
        "processType": "A16",    # Realised
        "outBiddingZone_Domain": country_code,
        "periodStart": period_start,
        "periodEnd": period_end,
        "securityToken": api_token
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)

    ns = {
        "ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"
    }

    period_nodes = root.findall(".//ns:Period", ns)
    if not period_nodes:
        raise ValueError("No <Period> nodes found in ENTSO-E response")

    records = []

    for period in period_nodes:
        start_time = period.find("ns:timeInterval/ns:start", ns).text
        start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

        for point in period.findall("ns:Point", ns):
            position = int(point.find("ns:position", ns).text)
            quantity = float(point.find("ns:quantity", ns).text)

            timestamp = start_time + pd.Timedelta(hours=position - 1)

            records.append({
                "datetime": timestamp,
                "load_MW": quantity
            })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("ENTSO-E returned an empty dataset")

    df = (
        df.sort_values("datetime")
          .drop_duplicates(subset=["datetime"])
          .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------------------
# Fetch and store 
# ---------------------------------------------------------------------
def fetch_entsoe_demand_and_store(
    country: str,
    country_code: str,
    start_year: int,
    end_year: int
):
    """
    Fetch and store ENTSO-E electricity demand as parquet files.
    Files are always overwritten.
    """

    load_dotenv()
    api_token = os.getenv("ENTSOE_API_TOKEN")
    if not api_token:
        raise ValueError("ENTSOE_API_TOKEN not found in environment")

    for year in range(start_year, end_year + 1):

        output_dir = (
            DATA_RAW_PATH
            / f"country={country}"
            / f"year={year}"
        )
        output_path = output_dir / "demand.parquet"

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[FETCH] ENTSO-E demand | {country} | {year}")

        try:
            df = fetch_entsoe_demand_one_year(
                year=year,
                country_code=country_code,
                api_token=api_token
            )

            df["country"] = country

            df.to_parquet(output_path, index=False)

            print(f"[SAVED] {output_path} | rows={len(df)}")

        except Exception as e:
            print(f"[ERROR] {country} {year} → {e}")

        time.sleep(random.uniform(0.3, 0.7))


# ---------------------------------------------------------------------
# Command line execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    fetch_entsoe_demand_and_store(
        country="FR",
        country_code="10YFR-RTE------C",
        start_year=2023,
        end_year=2024
    )
