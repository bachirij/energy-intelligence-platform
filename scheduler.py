"""
Scheduler — Automated Real-Time Pipeline
-----------------------------------------

Runs the real-time ingestion pipeline automatically every hour.

The scheduler triggers get_realtime_data.py at minute 5 of every hour
(e.g. 09:05, 10:05, 11:05...). The 5-minute offset gives ENTSO-E time
to publish the previous hour's data before we fetch it.

Usage:
------
    python scheduler.py                  # start with default config
    python scheduler.py --country FR     # explicit country
    python scheduler.py --interval 30    # run every 30 minutes (for testing)

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
# Centralized config (mirrors main.py)
# ---------------------------------------------------------------------

COUNTRIES = {
    "FR": {
        "entsoe_code": "10YFR-RTE------C",
        "latitude": 48.8534,
        "longitude": 2.3488,
    }
}


# ---------------------------------------------------------------------
# Import project module
# ---------------------------------------------------------------------

def import_realtime_module():
    """
    Import the real-time ingestion module.
    Done at runtime to show a clear error if the module is missing.
    """
    try:
        from src.ingestion.get_realtime_data import fetch_and_store_realtime
        return fetch_and_store_realtime
    except ImportError as e:
        logger.error(f"Failed to import get_realtime_data: {e}")
        logger.error("-> Make sure you run scheduler.py from the project root.")
        sys.exit(1)


# ---------------------------------------------------------------------
# Job: one execution of the real-time pipeline
# ---------------------------------------------------------------------

def realtime_job(country: str):
    """
    Single execution of the real-time pipeline.
    Called automatically by the scheduler at each trigger.
    """
    fetch_and_store_realtime = import_realtime_module()
    meta = COUNTRIES[country]

    logger.info(f"Job started | country={country} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    try:
        fetch_and_store_realtime(
            country=country,
            country_code=meta["entsoe_code"],
            latitude=meta["latitude"],
            longitude=meta["longitude"],
        )
        logger.info("Job completed successfully.")

    except Exception as e:
        logger.error(f"Job failed: {e}")
        raise  # re-raise so APScheduler catches it as EVENT_JOB_ERROR


# ---------------------------------------------------------------------
# Event listeners (logging job success / failure)
# ---------------------------------------------------------------------

def on_job_executed(event):
    logger.info(f"Next run scheduled at: {event.scheduled_run_time}")


def on_job_error(event):
    logger.error(f"Job raised an exception: {event.exception}")
    logger.error("The scheduler will retry at the next scheduled time.")


# ---------------------------------------------------------------------
# Scheduler setup and start
# ---------------------------------------------------------------------

def start_scheduler(country: str, interval_minutes: int = None):
    """
    Initialize and start the APScheduler.

    Parameters
    ----------
    country : str
        Country code to process (e.g. "FR")
    interval_minutes : int or None
        If set, run every N minutes (useful for testing).
        If None, run every hour at minute 5 (production mode).
    """
    if country not in COUNTRIES:
        logger.error(f"Country '{country}' not supported. Available: {list(COUNTRIES.keys())}")
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="UTC")

    # Register event listeners
    scheduler.add_listener(on_job_executed, EVENT_JOB_EXECUTED)
    scheduler.add_listener(on_job_error, EVENT_JOB_ERROR)

    # Schedule the job
    if interval_minutes:
        # Testing mode: run every N minutes
        scheduler.add_job(
            func=realtime_job,
            trigger="interval",
            minutes=interval_minutes,
            kwargs={"country": country},
            id="realtime_pipeline",
            name=f"Real-time pipeline ({country})",
            next_run_time=datetime.now(timezone.utc),  # run immediately on start
        )
        logger.info(f"Scheduler started in test mode — every {interval_minutes} min | country={country}")
    else:
        # Production mode: run every hour at minute 5
        # e.g. 09:05, 10:05, 11:05 ...
        scheduler.add_job(
            func=realtime_job,
            trigger="cron",
            minute=5,
            kwargs={"country": country},
            id="realtime_pipeline",
            name=f"Real-time pipeline ({country})",
        )
        logger.info(f"Scheduler started in production mode — every hour at :05 | country={country}")

    # Summary
    print("=" * 60)
    print("  energy-intelligence-platform — Scheduler")
    print("=" * 60)
    print(f"  Country  : {country}")
    print(f"  Mode     : {'test every ' + str(interval_minutes) + ' min' if interval_minutes else 'production — hourly at :05'}")
    print(f"  Started  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("  Stop     : Ctrl+C")
    print("=" * 60)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user (Ctrl+C).")
        print("\nScheduler stopped.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="energy-intelligence-platform — Automated real-time pipeline scheduler",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python scheduler.py                        # production mode, every hour at :05
  python scheduler.py --country FR           # explicit country
  python scheduler.py --interval 2           # test mode, every 2 minutes
        """
    )

    parser.add_argument(
        "--country",
        default="FR",
        choices=list(COUNTRIES.keys()),
        help="Country code to process (default: FR)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        dest="interval_minutes",
        metavar="MINUTES",
        help=(
            "Run every N minutes instead of hourly.\n"
            "Useful for testing without waiting a full hour.\n"
            "Example: --interval 2"
        )
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