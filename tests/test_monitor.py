"""
tests/test_monitor.py

Unit tests for src/monitoring/monitor.py.
Tests the pure logic (drift summary extraction, alerts) without real I/O.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers: synthetic DataFrames for testing
# ---------------------------------------------------------------------------

def make_ref_df(n=200) -> pd.DataFrame:
    """Reference DataFrame with stable distribution."""
    np.random.seed(42)
    return pd.DataFrame({
        "load_t-1":       np.random.normal(55000, 5000, n),
        "load_t-24":      np.random.normal(55000, 5000, n),
        "load_t-168":     np.random.normal(55000, 5000, n),
        "temperature_2m": np.random.normal(12, 8, n),
        "hour":           np.tile(np.arange(24), n // 24 + 1)[:n],
        "day_of_week":    np.random.randint(0, 7, n),
        "is_weekday":     np.random.randint(0, 2, n),
        "week_of_year":   np.random.randint(1, 53, n),
        "temperature_t-24": np.random.normal(12, 8, n),
    })


def make_cur_df_no_drift(n=50) -> pd.DataFrame:
    """Current DataFrame without drift (same distribution as reference)."""
    np.random.seed(99)
    return pd.DataFrame({
        "load_t-1":       np.random.normal(55000, 5000, n),
        "load_t-24":      np.random.normal(55000, 5000, n),
        "load_t-168":     np.random.normal(55000, 5000, n),
        "temperature_2m": np.random.normal(12, 8, n),
        "hour":           np.tile(np.arange(24), n // 24 + 1)[:n],
        "day_of_week":    np.random.randint(0, 7, n),
        "is_weekday":     np.random.randint(0, 2, n),
        "week_of_year":   np.random.randint(1, 53, n),
        "temperature_t-24": np.random.normal(12, 8, n),
    })


def make_cur_df_with_drift(n=50) -> pd.DataFrame:
    """Current DataFrame with strong drift on load_t-1 and temperature_2m."""
    np.random.seed(7)
    return pd.DataFrame({
        "load_t-1":       np.random.normal(80000, 3000, n),   # drift : mean very different from reference
        "load_t-24":      np.random.normal(55000, 5000, n),   # no drift
        "load_t-168":     np.random.normal(55000, 5000, n),   # no drift
        "temperature_2m": np.random.normal(35, 3, n),         # drift : heatwave simulated
        "hour":           np.tile(np.arange(24), n // 24 + 1)[:n],
        "day_of_week":    np.random.randint(0, 7, n),
        "is_weekday":     np.random.randint(0, 2, n),
        "week_of_year":   np.random.randint(1, 53, n),
        "temperature_t-24": np.random.normal(35, 3, n),
    })


FEATURE_COLS = [
    "load_t-1", "load_t-24", "load_t-168",
    "temperature_2m", "hour", "day_of_week",
    "is_weekday", "week_of_year", "temperature_t-24",
]


# ---------------------------------------------------------------------------
# Tests : compute_drift / _extract_drift_summary
# ---------------------------------------------------------------------------

class TestComputeDrift:

    def test_summary_contains_all_feature_cols(self):
        """The summary must contain an entry for each analyzed feature."""
        from monitor import compute_drift
        ref = make_ref_df()
        cur = make_cur_df_no_drift()
        summary, _ = compute_drift(ref, cur, FEATURE_COLS)
        for col in FEATURE_COLS:
            assert col in summary, f"Feature missing from summary : {col}"

    def test_summary_fields_structure(self):
        """Each entry in the summary must have drift_detected, p_value, method, threshold."""
        from monitor import compute_drift
        ref = make_ref_df()
        cur = make_cur_df_no_drift()
        summary, _ = compute_drift(ref, cur, FEATURE_COLS)
        for col, result in summary.items():
            assert "drift_detected" in result
            assert "p_value" in result
            assert "method" in result
            assert "threshold" in result
            assert isinstance(result["drift_detected"], bool)
            assert 0.0 <= result["p_value"] <= 1.0

    def test_drift_detected_when_distribution_shifts(self):
        """Features with very different distributions must be detected in drift."""
        from monitor import compute_drift
        ref = make_ref_df()
        cur = make_cur_df_with_drift()
        summary, _ = compute_drift(ref, cur, FEATURE_COLS)
        assert summary["load_t-1"]["drift_detected"] is True
        assert summary["temperature_2m"]["drift_detected"] is True

    def test_no_drift_when_distribution_stable(self):
        """Features with same distribution should not be detected in drift."""
        from monitor import compute_drift
        ref = make_ref_df()
        cur = make_cur_df_no_drift()
        summary, _ = compute_drift(ref, cur, FEATURE_COLS)
        # load_t-24 is stable in make_cur_df_no_drift, so should not be detected in drift
        assert summary["load_t-24"]["drift_detected"] is False

    def test_raw_dict_is_returned(self):
        """The raw_dict Evidently must be returned and contain metric_results."""
        from monitor import compute_drift
        ref = make_ref_df()
        cur = make_cur_df_no_drift()
        _, raw_dict = compute_drift(ref, cur, FEATURE_COLS)
        assert "metric_results" in raw_dict
        assert len(raw_dict["metric_results"]) > 0


# ---------------------------------------------------------------------------
# Tests : log_drift_alerts
# ---------------------------------------------------------------------------

class TestLogDriftAlerts:

    def test_returns_false_when_no_critical_drift(self):
        """Returns False if no critical feature is in drift."""
        from monitor import log_drift_alerts
        summary = {
            "load_t-1":       {"drift_detected": False, "p_value": 0.42, "method": "K-S p_value", "threshold": 0.05},
            "load_t-24":      {"drift_detected": False, "p_value": 0.31, "method": "K-S p_value", "threshold": 0.05},
            "temperature_2m": {"drift_detected": False, "p_value": 0.18, "method": "K-S p_value", "threshold": 0.05},
        }
        result = log_drift_alerts(summary)
        assert result is False

    def test_returns_true_when_critical_feature_drifts(self):
        """Returns True if at least one critical feature is in drift."""
        from monitor import log_drift_alerts
        summary = {
            "load_t-1":       {"drift_detected": True,  "p_value": 0.002, "method": "K-S p_value", "threshold": 0.05},
            "load_t-24":      {"drift_detected": False, "p_value": 0.31,  "method": "K-S p_value", "threshold": 0.05},
            "temperature_2m": {"drift_detected": False, "p_value": 0.18,  "method": "K-S p_value", "threshold": 0.05},
        }
        result = log_drift_alerts(summary)
        assert result is True

    def test_warn_logged_for_drifted_critical_feature(self):
        """A [WARN] message must be logged for each critical feature in drift."""
        from monitor import log_drift_alerts
        import logging
        summary = {
            "load_t-1":       {"drift_detected": True,  "p_value": 0.001, "method": "K-S p_value", "threshold": 0.05},
            "load_t-24":      {"drift_detected": False, "p_value": 0.40,  "method": "K-S p_value", "threshold": 0.05},
            "temperature_2m": {"drift_detected": True,  "p_value": 0.003, "method": "K-S p_value", "threshold": 0.05},
        }
        with patch("monitor.logger") as mock_logger:
            log_drift_alerts(summary)
            warn_calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("load_t-1" in c for c in warn_calls)
            assert any("temperature_2m" in c for c in warn_calls)


# ---------------------------------------------------------------------------
# Tests : save_monitoring_report
# ---------------------------------------------------------------------------

class TestSaveMonitoringReport:

    def test_file_created_with_correct_name(self, tmp_path):
        """The JSON file must be created with the name YYYY-MM-DD_HH.json."""
        from monitor import save_monitoring_report
        summary = {"load_t-1": {"drift_detected": False, "p_value": 0.42, "method": "K-S p_value", "threshold": 0.05}}
        raw_dict = {"metric_results": {}}
        ts = datetime(2026, 4, 11, 14, 0, 0)

        with patch("monitor.PROJECT_ROOT", tmp_path):
            output_path = save_monitoring_report(summary, raw_dict, ts)

        assert output_path.name == "2026-04-11_14.json"
        assert output_path.exists()

    def test_json_contains_required_keys(self, tmp_path):
        """The saved JSON must contain timestamp, drift_summary and evidently_raw."""
        import json
        from monitor import save_monitoring_report
        summary = {"load_t-1": {"drift_detected": False, "p_value": 0.42, "method": "K-S p_value", "threshold": 0.05}}
        raw_dict = {"metric_results": {}}
        ts = datetime(2026, 4, 11, 14, 0, 0)

        with patch("monitor.PROJECT_ROOT", tmp_path):
            output_path = save_monitoring_report(summary, raw_dict, ts)

        with open(output_path) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "drift_summary" in data
        assert "evidently_raw" in data
        assert "reference_year" in data
        