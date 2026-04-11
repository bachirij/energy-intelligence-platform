import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.build_features import _compute_features


# -----------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------
def make_continuous_df(n_hours: int = 200) -> pd.DataFrame:
    """
    Build a minimal continuous DataFrame with n_hours rows.
    Needs at least 168 rows to compute load_t-168.
    """
    base = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    datetimes = [base + pd.Timedelta(hours=i) for i in range(n_hours)]
    return pd.DataFrame({
        "datetime": datetimes,
        "load_MW": [50000.0 + i * 10 for i in range(n_hours)],
        "temperature_2m": [5.0 + i * 0.01 for i in range(n_hours)],
    })


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
def test_load_t_minus_168_is_correct():
    """
    load_t-168 at row i must equal load_MW at row i-168.
    """
    df = make_continuous_df(n_hours=200)
    result = _compute_features(df, country="FR")

    # Row 168 : load_t-168 should be load_MW at row 0
    row = result.iloc[168]
    assert row["load_t-168"] == pytest.approx(df["load_MW"].iloc[0]), \
        "load_t-168 at row 168 should equal load_MW at row 0"


def test_target_is_next_hour_load():
    """
    target_load_t+1 at row i must equal load_MW at row i+1.
    """
    df = make_continuous_df(n_hours=200)
    result = _compute_features(df, country="FR")

    for i in range(10):
        expected = df["load_MW"].iloc[i + 1]
        actual = result["target_load_t+1"].iloc[i]
        assert actual == pytest.approx(expected), \
            f"target_load_t+1 at row {i} should equal load_MW at row {i+1}"


def test_is_weekday_zero_on_sunday():
    """
    is_weekday must be 0 on Sundays.
    2026-01-04 is a Sunday.
    """
    df = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-01-04 10:00:00", tz="UTC")],
        "load_MW": [50000.0],
        "temperature_2m": [5.0],
    })
    result = _compute_features(df, country="FR")
    assert result["is_weekday"].iloc[0] == 0, "Sunday should have is_weekday=0"


def test_is_weekday_zero_on_french_holiday():
    """
    is_weekday must be 0 on French public holidays, even if it falls on a weekday.
    2026-07-14 is Bastille Day (Tuesday).
    """
    df = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-07-14 10:00:00", tz="UTC")],
        "load_MW": [50000.0],
        "temperature_2m": [5.0],
    })
    result = _compute_features(df, country="FR")
    assert result["is_weekday"].iloc[0] == 0, \
        "Bastille Day (Tuesday) should have is_weekday=0"


def test_is_weekday_one_on_regular_monday():
    """
    is_weekday must be 1 on a regular Monday (not a holiday).
    2026-01-05 is a Monday.
    """
    df = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-01-05 10:00:00", tz="UTC")],
        "load_MW": [50000.0],
        "temperature_2m": [5.0],
    })
    result = _compute_features(df, country="FR")
    assert result["is_weekday"].iloc[0] == 1, "Regular Monday should have is_weekday=1"


def test_last_row_has_no_target():
    """
    The last row must have NaN as target (no h+1 to predict).
    dropna() in the pipeline removes it — verified here at the source.
    """
    df = make_continuous_df(n_hours=10)
    result = _compute_features(df, country="FR")
    assert pd.isna(result["target_load_t+1"].iloc[-1]), \
        "Last row should have NaN target (no future hour available)"