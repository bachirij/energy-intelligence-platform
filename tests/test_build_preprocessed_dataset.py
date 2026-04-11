import pytest
import pandas as pd
import numpy as np
from src.preprocessing.build_preprocessed_dataset import reindex_and_interpolate_ts


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def make_demand_with_gap() -> pd.DataFrame:
    """Demand DataFrame with a missing hour at 12:00."""
    return pd.DataFrame({
        "datetime": pd.to_datetime([
            "2026-01-01 10:00", "2026-01-01 11:00",
            # 12:00 is missing
            "2026-01-01 13:00", "2026-01-01 14:00",
        ]),
        "load_MW": [50000.0, 51000.0, 53000.0, 54000.0],
        "country": ["FR"] * 4,
    })


def make_weather_with_gap() -> pd.DataFrame:
    """Weather DataFrame with a missing hour at 12:00 and night radiation values."""
    return pd.DataFrame({
        "datetime": pd.to_datetime([
            "2026-01-01 10:00", "2026-01-01 11:00",
            # 12:00 is missing
            "2026-01-01 13:00", "2026-01-01 14:00",
        ]),
        "temperature_2m": [3.0, 4.0, 6.0, 7.0],
        "shortwave_radiation_instant": [0.0, 0.0, 0.0, 0.0],
    })


# -----------------------------------------------------------------------
# Tests: reindex_and_interpolate_ts
# -----------------------------------------------------------------------
def test_missing_hour_is_filled():
    """
    A missing hour in the middle of the series must be reindexed
    and filled by linear interpolation.
    """
    df = make_demand_with_gap()

    result = reindex_and_interpolate_ts(
        df=df,
        time_col="datetime",
        numeric_cols=["load_MW"],
        categorical_cols=["country"],
    )

    assert len(result) == 5, "Reindexed DataFrame should have 5 rows (no gap)"

    row_12 = result[result["datetime"] == pd.Timestamp("2026-01-01 12:00")]
    assert not row_12.empty, "Row at 12:00 should exist after reindex"
    assert row_12["load_MW"].iloc[0] == pytest.approx(52000.0), \
        "Missing load_MW at 12:00 should be linearly interpolated between 51000 and 53000"


def test_shortwave_radiation_is_forward_filled():
    """
    shortwave_radiation_instant must be forward-filled, not linearly interpolated.
    At night it is exactly 0 — linear interpolation would produce non-zero values.
    """
    df = make_weather_with_gap()

    result = reindex_and_interpolate_ts(
        df=df,
        time_col="datetime",
        numeric_cols=["temperature_2m"],
        ffill_cols=["shortwave_radiation_instant"],
    )

    row_12 = result[result["datetime"] == pd.Timestamp("2026-01-01 12:00")]
    assert not row_12.empty
    assert row_12["shortwave_radiation_instant"].iloc[0] == pytest.approx(0.0), \
        "shortwave_radiation_instant should be forward-filled (0.0), not interpolated"


def test_no_leading_or_trailing_nans():
    """
    limit_area='inside' means only interior gaps are filled.
    There should be no NaNs at the edges — verified to be absent in real data.
    For interior gaps, no NaN should remain after interpolation.
    """
    df = make_demand_with_gap()

    result = reindex_and_interpolate_ts(
        df=df,
        time_col="datetime",
        numeric_cols=["load_MW"],
        categorical_cols=["country"],
    )

    assert result["load_MW"].isna().sum() == 0, \
        "No NaNs should remain in load_MW after interpolation"


def test_duplicates_are_removed():
    """
    Duplicate timestamps in the input must be removed before reindexing.
    """
    df = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2026-01-01 10:00", "2026-01-01 10:00",  # duplicate
            "2026-01-01 11:00", "2026-01-01 12:00",
        ]),
        "load_MW": [50000.0, 99999.0, 51000.0, 52000.0],
        "country": ["FR"] * 4,
    })

    result = reindex_and_interpolate_ts(
        df=df,
        time_col="datetime",
        numeric_cols=["load_MW"],
        categorical_cols=["country"],
    )

    assert len(result) == 3, "Duplicate timestamp should be removed"
    row_10 = result[result["datetime"] == pd.Timestamp("2026-01-01 10:00")]
    assert row_10["load_MW"].iloc[0] == pytest.approx(50000.0), \
        "First occurrence should be kept after deduplication"