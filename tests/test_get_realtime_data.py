import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.ingestion.get_realtime_data import build_realtime_snapshot, save_realtime_snapshot


# -----------------------------------------------------------------------
# Helpers: build minimal test DataFrames
# -----------------------------------------------------------------------
def make_demand(hours: list[int]) -> pd.DataFrame:
    """Create a minimal demand DataFrame with load_MW for given hour offsets."""
    base = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
    return pd.DataFrame({
        "datetime": [base + pd.Timedelta(hours=h) for h in hours],
        "load_MW": [50000.0 + h * 100 for h in hours],
    })


def make_weather(hours: list[int]) -> pd.DataFrame:
    """Create a minimal weather DataFrame for given hour offsets."""
    base = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
    return pd.DataFrame({
        "datetime": [base + pd.Timedelta(hours=h) for h in hours],
        "temperature_2m": [5.0 + h * 0.1 for h in hours],
    })


# -----------------------------------------------------------------------
# Tests: build_realtime_snapshot
# -----------------------------------------------------------------------
def test_weather_shift_minus_one_hour():
    """
    Weather at t+1 must be stored at row t after the -1h shift.
    If Open-Meteo returns temperature for 11:00 and 12:00,
    after the shift they should appear at rows 10:00 and 11:00.
    """
    df_demand = make_demand(hours=[0, 1, 2])   # 10:00, 11:00, 12:00
    df_weather = make_weather(hours=[1, 2])    # weather originally at 11:00, 12:00

    result = build_realtime_snapshot(df_demand, df_weather)

    # After -1h shift, weather originally at 11:00 should be on row 10:00
    row_t = result[result["datetime"] == pd.Timestamp("2026-01-01 10:00:00", tz="UTC")]
    assert not row_t.empty, "Row at t=10:00 should exist"
    assert row_t["temperature_2m"].iloc[0] == pytest.approx(5.1), \
        "temperature at t=10:00 should be the weather value originally at 11:00"


def test_left_join_keeps_demand_rows_without_weather():
    """
    Rows where demand exists but weather is NaN must be kept (left join).
    This happens for historical demand rows outside the weather forecast window.
    """
    df_demand = make_demand(hours=[0, 1, 2, 3])  # 4 demand rows
    df_weather = make_weather(hours=[1, 2])       # weather only covers 2 hours

    result = build_realtime_snapshot(df_demand, df_weather)

    assert len(result) == 4, "All 4 demand rows must be present after left join"


# -----------------------------------------------------------------------
# Tests: save_realtime_snapshot
# -----------------------------------------------------------------------
def test_rolling_window_drops_old_rows():
    """
    Rows older than rolling_window_hours must be dropped on save.
    """
    now = datetime.now(timezone.utc)

    df_new = pd.DataFrame({
        "datetime": [
            now - timedelta(hours=300),  # old - must be dropped
            now - timedelta(hours=100),  # recent - must be kept
            now - timedelta(hours=1),    # recent - must be kept
        ],
        "load_MW": [50000.0, 51000.0, 52000.0],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch REALTIME_PATH to point to our temp directory
        with patch("src.ingestion.get_realtime_data.REALTIME_PATH", Path(tmpdir)):
            save_realtime_snapshot(df_new, country="FR", rolling_window_hours=192)

            saved = pd.read_parquet(
                Path(tmpdir) / "country=FR" / "realtime.parquet"
            )

    assert len(saved) == 2, "Only the 2 recent rows should be kept"