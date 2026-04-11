"""
src/monitoring/monitor.py

Detect the drift between the features of the realtime snapshot and the reference distribution (featured data 2024).
Saves a timestamped JSON report in data/monitoring/ and logs a [WARN] alert if a critical feature drifts.

"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from src.modeling.config import COUNTRY
from evidently import Dataset, DataDefinition
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently import Report

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Critical features: [WARN] alert if any of them drift
CRITICAL_FEATURES = ["load_t-1", "load_t-24", "load_t-168"]

# All features to monitor (subset of FEATURE_COLS, excluding weather features with too many NaNs in the snapshot)
MONITORABLE_FEATURES = [
    "load_t", "load_t-1", "load_t-24", "load_t-168",
    "hour", "day_of_week", "is_weekday", "week_of_year",
]

# Standard KS threshold: p-value < 0.05 → drift detected
DRIFT_THRESHOLD = 0.05

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load data and compute features
# ---------------------------------------------------------------------------

def load_reference_features() -> pd.DataFrame:
    """
    Load the reference features from featured/country=FR/year=2024/.
    Year 2024 = most recent distribution seen during training.
    """
    ref_path = PROJECT_ROOT / "data" / "featured" / "country=FR" / "year=2024" / "load_forecasting_features.parquet"
    if not ref_path.exists():
        raise ValueError(f"[ERROR] Reference data not found : {ref_path}")
    df = pd.read_parquet(ref_path)
    logger.info(f"[FETCH] Reference data loaded : {len(df)} rows from {ref_path}")
    return df


def load_realtime_features() -> pd.DataFrame:
    """
    Load the realtime snapshot and compute the load and calendar features.
    Weather features are excluded because the snapshot only contains 2h of weather forecast, the temperature columns would be NaN on ~99% of the rows.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    realtime_path = PROJECT_ROOT / "data" / "realtime" / "country=FR" / "realtime.parquet"
    if not realtime_path.exists():
        raise ValueError(f"[ERROR] Realtime data not found : {realtime_path}")

    df = pd.read_parquet(realtime_path)
    logger.info(f"[FETCH] Realtime snapshot loaded : {len(df)} rows from {realtime_path}")
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    df["load_t-1"]   = df["load_MW"].shift(1)
    df["load_t-24"]  = df["load_MW"].shift(24)
    df["load_t-168"] = df["load_MW"].shift(24 * 7)

    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekday"]  = (df["day_of_week"] < 5).astype(int)
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    df.loc[df["week_of_year"] == 53, "week_of_year"] = 52

    df = df.rename(columns={"load_MW": "load_t"})
    df = df.dropna(subset=["load_t-1", "load_t-24", "load_t-168"])

    logger.info(f"[PROCESS] Realtime features computed : {len(df)} rows after dropna")
    return df


# ---------------------------------------------------------------------------
# Drift computation with Evidently
# ---------------------------------------------------------------------------

def _build_evidently_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[Report, dict]:
    """
    Construct and run an Evidently report on the common feature_cols.
    Returns the Evidently Report and the raw dump_dict.
    """
    # Keeps only the features that exist in both datasets
    cols = [c for c in feature_cols if c in ref_df.columns and c in cur_df.columns]
    if not cols:
        raise ValueError("[ERROR] No common features found between reference and current data.")

    definition = DataDefinition(numerical_columns=cols)
    ref_ds = Dataset.from_pandas(ref_df[cols].reset_index(drop=True), data_definition=definition)
    cur_ds = Dataset.from_pandas(cur_df[cols].reset_index(drop=True), data_definition=definition)

    metrics = [DriftedColumnsCount()] + [ValueDrift(column=c) for c in cols]
    report = Report(metrics)
    result = report.run(reference_data=ref_ds, current_data=cur_ds)

    return result, result.dump_dict()


def _extract_drift_summary(raw_dict: dict, feature_cols: list[str]) -> dict:
    """
    Extract a readable summary from the Evidently dump_dict.
    Format: {feature: {drift_detected, p_value, method}}
    Only for the features in feature_cols.
    Uses the ValueDrift results to determine drift_detected based on p_value < DRIFT_THRESHOLD.
    """
    summary = {}

    for metric_result in raw_dict["metric_results"].values():
        params = metric_result.get("metric_value_location", {}).get("metric", {}).get("params", {})

        # ValueDrift : type evidently:metric_v2:ValueDrift
        if params.get("type") == "evidently:metric_v2:ValueDrift":
            col = params.get("column")
            p_value = metric_result.get("value")
            method = params.get("method", "unknown")
            if col and p_value is not None:
                summary[col] = {
                    "drift_detected": bool(p_value < DRIFT_THRESHOLD),
                    "p_value": round(p_value, 6),
                    "method": method,
                    "threshold": DRIFT_THRESHOLD,
                }

    return summary


def compute_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict, dict]:
    """
    Compute the drift between ref_df and cur_df on feature_cols.
    Returns (summary per feature, complete Evidently raw_dict).
    """
    _, raw_dict = _build_evidently_report(ref_df, cur_df, feature_cols)
    summary = _extract_drift_summary(raw_dict, feature_cols)
    return summary, raw_dict


# ---------------------------------------------------------------------------
# Save report and log alerts
# ---------------------------------------------------------------------------

def save_monitoring_report(summary: dict, raw_dict: dict, timestamp: datetime) -> Path:
    """
    Saves the monitoring report in data/monitoring/YYYY-MM-DD_HH.json.
    Contains the readable summary + the raw Evidently dump.
    """
    output_dir = PROJECT_ROOT / "data" / "monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = timestamp.strftime("%Y-%m-%d_%H") + ".json"
    output_path = output_dir / filename

    report = {
        "timestamp": timestamp.isoformat(),
        "reference_year": 2024,
        "n_features_analyzed": len(summary),
        "drift_summary": summary,
        "evidently_raw": raw_dict,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"[SAVED] Report monitoring saved : {output_path}")
    return output_path


def log_drift_alerts(summary: dict) -> bool:
    """
    Logs a [WARN] alert for each critical feature in drift.
    Returns True if at least one critical feature is in drift.
    """
    critical_in_drift = []

    for feature in CRITICAL_FEATURES:
        if feature not in summary:
            logger.warning(f"[WARN] Critical feature absent from report : {feature}")
            continue
        result = summary[feature]
        if result["drift_detected"]:
            critical_in_drift.append(feature)
            logger.warning(
                f"[WARN] Drift detected on critical feature '{feature}' "
                f"(p_value={result['p_value']:.4f} < {DRIFT_THRESHOLD})"
            )

    if not critical_in_drift:
        logger.info("[PROCESS] No drift detected on critical features.")

    return len(critical_in_drift) > 0


# ---------------------------------------------------------------------------
# Principal pipeline
# ---------------------------------------------------------------------------

def run_monitoring() -> Path:
    """
    Complete pipeline for drift monitoring:
    1. Load the reference (featured 2024) and the realtime snapshot
    2. Compute the drift with Evidently
    3. Save the timestamped JSON report
    4. Log alerts on critical features in drift
    Returns the path of the saved report.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from modeling.config import FEATURE_COLS

    timestamp = datetime.utcnow()
    logger.info(f"[PROCESS] Starting drift monitoring — {timestamp.strftime('%Y-%m-%d %H:%M UTC')}")

    ref_df = load_reference_features()
    cur_df = load_realtime_features()

    logger.info(f"[PROCESS] Computing drift on {len(FEATURE_COLS)} features...")
    summary, raw_dict = compute_drift(ref_df, cur_df, MONITORABLE_FEATURES)
    logger.info(
    "[WARN] Weather features are excluded from monitoring (temperature_t, temperature_t-24) : "
    "the realtime snapshot only contains 2h of weather forecast.")

    n_drifted = sum(1 for v in summary.values() if v["drift_detected"])
    logger.info(f"[PROCESS] Global result : {n_drifted}/{len(summary)} features in drift")

    log_drift_alerts(summary)
    output_path = save_monitoring_report(summary, raw_dict, timestamp)

    return output_path


if __name__ == "__main__":
    run_monitoring()