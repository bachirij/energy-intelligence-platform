import os
import time
import random
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from dotenv import load_dotenv


BASE_URL = "https://web-api.tp.entsoe.eu/api"


def fetch_entsoe_demand_one_year(
    year: int,
    country_code: str,
    api_token: str
) -> pd.DataFrame:
    """
    Fetch hourly actual electricity load for one year from ENTSO-E.
    """

    period_start = f"{year}01010000"
    period_end = f"{year + 1}01010000"

    params = {
        "documentType": "A65",
        "processType": "A16",
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

    period_node = root.find(".//ns:Period", ns)
    if period_node is None:
        raise ValueError("No Period node found")

    start_time = period_node.find("ns:timeInterval/ns:start", ns).text
    start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

    data = []
    for point in period_node.findall("ns:Point", ns):
        position = int(point.find("ns:position", ns).text)
        quantity = float(point.find("ns:quantity", ns).text)
        timestamp = start_time + pd.Timedelta(hours=position - 1)

        data.append({
            "datetime": timestamp,
            "load_MW": quantity
        })

    return pd.DataFrame(data)


def fetch_entsoe_demand_and_store(
    country: str,
    country_code: str,
    start_year: int,
    end_year: int,
    base_path: str = "data/raw/electricity_demand"
):
    """
    Fetch and store ENTSO-E electricity demand data as parquet files
    partitioned by country and year.
    """

    load_dotenv()
    api_token = os.getenv("ENTSOE_API_TOKEN")
    if not api_token:
        raise ValueError("ENTSOE_API_TOKEN not found")

    for year in range(start_year, end_year + 1):

        output_dir = f"{base_path}/country={country}/year={year}"
        output_path = f"{output_dir}/demand.parquet"
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(output_path):
            print(f"[SKIP] {country} {year} already exists")
            continue

        print(f"[FETCH] ENTSO-E demand | {country} | {year}")

        try:
            df = fetch_entsoe_demand_one_year(year, country_code, api_token)
            df["country"] = country
            df = df.sort_values("datetime")

            df.to_parquet(output_path, index=False)
            print(f"[SAVED] {output_path}")

        except Exception as e:
            print(f"[ERROR] {country} {year} → {e}")

        time.sleep(random.uniform(0.3, 0.7))


if __name__ == "__main__":
    fetch_entsoe_demand_and_store(
        country="FR",
        country_code="10YFR-RTE------C",
        start_year=2023,
        end_year=2024
    )
