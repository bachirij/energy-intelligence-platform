import os
import time
import random
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from dotenv import load_dotenv


BASE_URL = "https://web-api.tp.entsoe.eu/api"


def get_entsoe_load(start_date: str, end_date: str, country_code: str = "10YFR-RTE------C") -> pd.DataFrame:
    """
    Retrieve historical hourly electricity load (Actual Total Load) from the ENTSO-E Transparency Platform.
    
    The function:
    - Splits the requested date range into yearly chunks (max 1-year per API call).
    - Fetches hourly total load data for each chunk.
    - Converts XML responses to pandas DataFrames.
    - Concatenates all results into a single DataFrame.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    country_code : str, default '10YFR-RTE------C'
        EIC code of the bidding zone (default = France).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'datetime' (pd.Timestamp, UTC)
        - 'load_MW' (float)
    """
    
    # Load API token from .env file
    load_dotenv()
    API_TOKEN = os.getenv("ENTSOE_API_TOKEN")
    if not API_TOKEN:
        raise ValueError("❌ API key not found in .env file (ENTSOE_API_TOKEN)")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    all_dfs = []

    # Split into yearly chunks (ENTSO-E API limit = 1 year)
    for year_start in pd.date_range(start, end, freq="YS"):
        year_end = min(year_start + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        print(f"Fetching: {year_start.strftime('%Y-%m-%d')} to {year_end.strftime('%Y-%m-%d')} ...")

        # Format dates for API (yyyyMMddHHmm)
        period_start = year_start.strftime("%Y%m%d0000")
        period_end = (year_end + pd.Timedelta(days=1)).strftime("%Y%m%d0000")  # include last day

        params = {
            "documentType": "A65",  # Actual Total Load
            "processType": "A16",   # Realised
            "outBiddingZone_Domain": country_code,
            "periodStart": period_start,
            "periodEnd": period_end,
            "securityToken": API_TOKEN
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            if "text/xml" not in response.headers.get("Content-Type", ""):
                print("⚠️ Unexpected response (non-XML), skipping this period.")
                continue

            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                print(f"⚠️ XML parsing error ({e}), skipping this period.")
                continue

            ns = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}
            period_node = root.find(".//ns:Period", ns)
            if period_node is None:
                print("⚠️ No 'Period' node found (no data available).")
                continue

            start_time = period_node.find("ns:timeInterval/ns:start", ns).text
            points = period_node.findall("ns:Point", ns)
            if not points:
                print("⚠️ No data points found in this period.")
                continue

            data = []
            for p in points:
                try:
                    pos = int(p.find("ns:position", ns).text)
                    quantity = float(p.find("ns:quantity", ns).text)
                    timestamp = datetime.fromisoformat(start_time.replace("Z", "+00:00")) + pd.Timedelta(hours=pos - 1)
                    data.append({"datetime": timestamp, "load_MW": quantity})
                except Exception as e:
                    print(f"⚠️ Invalid data point ({e}), skipped.")
                    continue

            if data:
                df_chunk = pd.DataFrame(data)
                all_dfs.append(df_chunk)

        except requests.exceptions.Timeout:
            print("❌ Request timed out, skipping this period.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

        # Random small delay to avoid hitting rate limits
        time.sleep(random.uniform(0.2, 0.6))

    if not all_dfs:
        raise ValueError("❌ No data could be retrieved for the requested period.")

    # Concatenate all chunks into one DataFrame
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sort_values("datetime").reset_index(drop=True)

    return full_df

# ---------------------- MAIN EXECUTION ----------------------
if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)

    # Example: fetch hourly load data from 2023 to 2025 for France
    df = get_entsoe_load("2023-01-01", "2025-01-31", country_code="10YFR-RTE------C")
    df.to_csv("../data/electricity_load_france_2023_2025.csv", index=False)
    print("✅ Data saved to '../data/electricity_load_france_2023_2025.csv'")
    
