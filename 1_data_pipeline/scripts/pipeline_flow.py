import os
from import_meteo import get_openmeteo_data
# from import_entsoe import get_entsoe_data  # future import for ENTSO-E

# -------------------------
# 1. Define paths
# -------------------------
# BASE_DIR points to the root of 1_data_pipeline
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  

# DATA_DIR is where all CSVs will be stored
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)  # create data folder if it doesn't exist

# -------------------------
# 2. Pipeline parameters
# -------------------------
START_DATE = "2025-01-01"
END_DATE   = "2025-01-02"
WEATHER_FILE = os.path.join(DATA_DIR, "weather_jan2025.csv")
# ENTSOE_FILE = os.path.join(DATA_DIR, "entsoe_jan2025.csv")  # placeholder for ENTSO-E

# -------------------------
# 3. Utility functions
# -------------------------
def save_csv(df, path):
    """
    Save a DataFrame to CSV and print a confirmation message.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    path : str
        Full path where the CSV will be saved.
    """
    df.to_csv(path, index=False)
    print(f"Saved CSV at {path}")

# -------------------------
# 4. Main pipeline
# -------------------------
def main():
    """
    Main function to run the data pipeline.

    Steps:
    1. Fetch weather data from Open-Meteo
    2. Save the weather data as CSV
    3. (Future) Fetch ENTSO-E data and save as CSV
    """
    # 4a. Fetch historical weather data
    df_weather = get_openmeteo_data(START_DATE, END_DATE)
    save_csv(df_weather, WEATHER_FILE)

    # 4b. Fetch ENTSO-E data (to be implemented later)
    # df_entsoe = get_entsoe_data(START_DATE, END_DATE)
    # save_csv(df_entsoe, ENTSOE_FILE)

    print("Pipeline finished successfully!")

# -------------------------
# 5. Entry point
# -------------------------
if __name__ == "__main__":
    # This allows the script to be run directly from terminal
    main()
