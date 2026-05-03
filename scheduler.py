"""
Scheduler — Automated Pipeline
--------------------------------

Two jobs run automatically:

1. Realtime job (every hour at :05)
   → fetch_and_store_realtime: maintains the 192h rolling parquet
     used by /predict and the real-time dashboard tab.

2. Daily job (every day at 06:00 UTC)
   → ingest + preprocess + features for the current year only.
     Refreshes the 2026 historical parquets as new days are published
     by ENTSO-E. Guard clauses in ingestion skip completed years
     (< current year) automatically.

Usage:
------
    python scheduler.py                  # start with default config
    python scheduler.py --country FR     # explicit country
    python scheduler.py --interval 30    # realtime every 30 min (testing)

Stop with Ctrl+C.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR


# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

COUNTRIES = {
    "FR": {
        "entsoe_code": "10YFR-RTE------C",
        "latitude": 48.8534,
        "longitude": 2.3488,
    }
}


# ---------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------

def import_realtime_module():
    """Import the real-time ingestion function."""
    try:
        from src.ingestion.get_realtime_data import fetch_and_store_realtime
        return fetch_and_store_realtime
    except ImportError as e:
        logger.error(f"Failed to import get_realtime_data: {e}")
        logger.error("-> Make sure you run scheduler.py from the project root.")
        sys.exit(1)


def import_pipeline_module():
    """Import the main pipeline orchestrator."""
    try:
        from main import run_pipeline
        return run_pipeline
    except ImportError as e:
        logger.error(f"Failed to import main.run_pipeline: {e}")
        logger.error("-> Make sure you run scheduler.py from the project root.")
        sys.exit(1)


# ---------------------------------------------------------------------
# Job 1 — Realtime (every hour at :05)
# ---------------------------------------------------------------------

def realtime_job(country: str):
    """
    Fetch the latest demand and weather forecast, update the rolling
    parquet, and run drift monitoring.
    Triggered every hour at minute 5.
    """
    fetch_and_store_realtime = import_realtime_module()
    meta = COUNTRIES[country]

    logger.info(
        f"[REALTIME] Job started | country={country} | "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    try:
        fetch_and_store_realtime(
            country=country,
            country_code=meta["entsoe_code"],
            latitude=meta["latitude"],
            longitude=meta["longitude"],
        )
        logger.info("[REALTIME] Job completed successfully.")

    except Exception as e:
        logger.error(f"[REALTIME] Job failed: {e}")
        raise


# ---------------------------------------------------------------------
# Job 2 — Daily historical refresh (every day at 06:00 UTC)
# ---------------------------------------------------------------------

def daily_job(country: str):
    """
    Refresh current-year historical parquets (ingest → preprocess → features).
    Triggered once per day at 06:00 UTC, after ENTSO-E has published
    overnight data with its usual 1-2h publication latency.

    Years before the current year are skipped automatically by the
    guard clauses in the ingestion functions.
    """
    run_pipeline = import_pipeline_module()
    current_year = datetime.now(timezone.utc).year

    logger.info(
        f"[DAILY] Job started | country={country} | year={current_year} | "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    try:
        run_pipeline(
            steps=["ingest", "preprocess", "features"],
            country=country,
            start_year=current_year,
            end_year=current_year,
        )
        logger.info("[DAILY] Job completed successfully.")

    except Exception as e:
        logger.error(f"[DAILY] Job failed: {e}")
        raise


# ---------------------------------------------------------------------
# Event listeners
# ---------------------------------------------------------------------

def on_job_executed(event):
    logger.info(f"[SCHEDULER] Job '{event.job_id}' executed successfully.")


def on_job_error(event):
    logger.error(f"[SCHEDULER] Job '{event.job_id}' raised an exception: {event.exception}")
    logger.error("[SCHEDULER] Will retry at next scheduled time.")


# ---------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------

def start_scheduler(country: str, interval_minutes: int = None):
    """
    Initialize and start the APScheduler with two jobs:
    - realtime_pipeline : every hour at :05 (or every N min in test mode)
    - daily_pipeline    : every day at 06:00 UTC (skipped in test mode)

    Parameters
    ----------
    country : str
        Country code to process (e.g. "FR")
    interval_minutes : int or None
        If set, run realtime job every N minutes (test mode).
        Daily job is disabled in test mode.
        If None, production mode: realtime at :05, daily at 06:00 UTC.
    """
    if country not in COUNTRIES:
        logger.error(f"Country '{country}' not supported. Available: {list(COUNTRIES.keys())}")
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="UTC")

    scheduler.add_listener(on_job_executed, EVENT_JOB_EXECUTED)
    scheduler.add_listener(on_job_error, EVENT_JOB_ERROR)

    if interval_minutes:
        # ---- Test mode ----
        # Realtime job every N minutes, runs immediately on start
        scheduler.add_job(
            func=realtime_job,
            trigger="interval",
            minutes=interval_minutes,
            kwargs={"country": country},
            id="realtime_pipeline",
            name=f"Realtime pipeline ({country})",
            next_run_time=datetime.now(timezone.utc),
        )
        logger.info(
            f"[SCHEDULER] Test mode - realtime every {interval_minutes} min | "
            f"daily job disabled | country={country}"
        )

    else:
        # ---- Production mode ----

        # Job 1: realtime every hour at minute 5
        scheduler.add_job(
            func=realtime_job,
            trigger="cron",
            minute=5,
            kwargs={"country": country},
            id="realtime_pipeline",
            name=f"Realtime pipeline ({country})",
        )

        # Job 2: daily refresh at 06:00 UTC
        scheduler.add_job(
            func=daily_job,
            trigger="cron",
            hour=6,
            minute=0,
            kwargs={"country": country},
            id="daily_pipeline",
            name=f"Daily historical refresh ({country})",
        )

        logger.info(
            f"[SCHEDULER] Production mode | "
            f"realtime at :05 every hour | "
            f"daily refresh at 06:00 UTC | "
            f"country={country}"
        )

    # Summary banner
    print("=" * 60)
    print("  energy-intelligence-platform - Scheduler")
    print("=" * 60)
    print(f"  Country   : {country}")
    if interval_minutes:
        print(f"  Mode      : TEST - realtime every {interval_minutes} min")
        print(f"  Daily job : disabled")
    else:
        print(f"  Mode      : PRODUCTION")
        print(f"  Realtime  : every hour at :05")
        print(f"  Daily     : every day at 06:00 UTC")
    print(f"  Started   : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("  Stop      : Ctrl+C")
    print("=" * 60)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("[SCHEDULER] Stopped by user (Ctrl+C).")
        print("\nScheduler stopped.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="energy-intelligence-platform - Automated pipeline scheduler",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python scheduler.py                   # production mode
  python scheduler.py --country FR      # explicit country
  python scheduler.py --interval 2      # test mode, realtime every 2 min
        """,
    )

    parser.add_argument(
        "--country",
        default="FR",
        choices=list(COUNTRIES.keys()),
        help="Country code to process (default: FR)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        dest="interval_minutes",
        metavar="MINUTES",
        help=(
            "Run realtime job every N minutes (test mode).\n"
            "Daily job is disabled in test mode.\n"
            "Example: --interval 2"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    start_scheduler(
        country=args.country,
        interval_minutes=args.interval_minutes,
    )