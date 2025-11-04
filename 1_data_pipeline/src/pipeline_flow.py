import os
from import_meteo import get_openmeteo_data
from import_entsoe import get_entsoe_load
from preprocess import preprocess_all

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
ENTSOE_FILE = os.path.join(DATA_DIR, "entsoe_jan2025.csv")  # placeholder for ENTSO-E

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
    df_weather = get_openmeteo_data(START_DATE, END_DATE)
    df_load = get_entsoe_load(START_DATE, END_DATE)

    df_inputs = preprocess_all(df_weather, df_load)
    save_csv(df_inputs, os.path.join(DATA_DIR, "preprocessed_inputs.csv"))

# -------------------------
# 5. Entry point
# -------------------------
if __name__ == "__main__":
    # Allows the script to be run directly from terminal
    main()
