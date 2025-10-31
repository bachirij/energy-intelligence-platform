import pandas as pd

def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess weather data"""

    # Convert 'datetime' to datetime and remove timezone information
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")

    # Sort the DateFrame by 'datetime'
    df = df.sort_values(by = "datetime").reset_index(drop=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").interpolate(method="time").reset_index()
    return df

