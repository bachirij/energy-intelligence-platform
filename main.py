"""
energy-intelligence-platform — Pipeline Orchestrator
=====================================================

Single entry point to run all or part of the data pipeline.

Usage:
------
    # Run all steps (ingest → preprocess → features → train)
    python main.py

    # Run ingestion only
    python main.py --steps ingest

    # Run preprocessing and feature engineering only
    python main.py --steps preprocess features

    # Train models only
    python main.py --steps train

    # Run one real-time ingestion cycle manually
    python main.py --steps realtime

    # Run over a specific year range
    python main.py --steps ingest preprocess features --start-year 2023 --end-year 2025

    # Display help
    python main.py --help

Notes:
------
- Steps ingest / preprocess / features use --start-year and --end-year.
- Step train ignores those args: temporal splits are managed by src/modeling/config.py
  and update automatically each year (test = current_year - 1, val = current_year - 2).
- Step realtime also ignores year args (fetches last 48h of data).
- Guard clauses in each step skip already-existing files automatically.
  Delete the relevant parquet files manually if you need to reprocess them.
"""

import argparse
import sys
import time
from datetime import datetime


# ---------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------

COUNTRIES = {
    "FR": {
        "entsoe_code": "10YFR-RTE------C",
        "latitude":    48.8534,
        "longitude":   2.3488,
    }
}

ALL_STEPS = ["ingest", "preprocess", "features", "train", "realtime"]

DEFAULT_COUNTRY    = "FR"
DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR   = datetime.now().year - 1  # most recent complete year


# ---------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------

def import_modules() -> dict:
    """
    Import all pipeline modules.

    Imports are centralised here (rather than at the top of the file)
    to surface a clear error message if a module or dependency is missing,
    instead of a raw Python traceback.
    """
    try:
        from src.ingestion.get_entsoe_demand import fetch_entsoe_demand_and_store
        from src.ingestion.get_openmeteo_weather import fetch_openmeteo_weather_and_store
        from src.ingestion.get_realtime_data import fetch_and_store_realtime
        from src.preprocessing.build_preprocessed_dataset import build_processed_dataset
        from src.feature_engineering.build_features import build_load_forecasting_features
        from src.modeling.train import run_training

        return {
            "fetch_entsoe_demand_and_store":     fetch_entsoe_demand_and_store,
            "fetch_openmeteo_weather_and_store": fetch_openmeteo_weather_and_store,
            "fetch_and_store_realtime":          fetch_and_store_realtime,
            "build_processed_dataset":           build_processed_dataset,
            "build_load_forecasting_features":   build_load_forecasting_features,
            "run_training":                      run_training,
        }

    except ImportError as e:
        print(f"\n[ERROR] Failed to import a module: {e}")
        print("  → Make sure you run main.py from the project root.")
        print("  → Example: python main.py --steps ingest\n")
        sys.exit(1)


# ---------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------

def step_ingest(country: str, start_year: int, end_year: int, modules: dict):
    """
    Step 1 — Raw data ingestion.
    Fetches electricity demand (ENTSO-E) and weather (Open-Meteo)
    for each year in [start_year, end_year].
    Already-existing parquet files are skipped automatically.
    """
    meta = COUNTRIES[country]

    print("\n--- ENTSO-E ingestion (electricity demand) ---")
    modules["fetch_entsoe_demand_and_store"](
        country=country,
        country_code=meta["entsoe_code"],
        start_year=start_year,
        end_year=end_year,
    )

    print("\n--- Open-Meteo ingestion (weather) ---")
    modules["fetch_openmeteo_weather_and_store"](
        country=country,
        latitude=meta["latitude"],
        longitude=meta["longitude"],
        start_year=start_year,
        end_year=end_year,
    )


def step_preprocess(country: str, start_year: int, end_year: int, modules: dict):
    """
    Step 2 — Preprocessing.
    Cleans, reindexes, interpolates and merges demand + weather
    for each year in [start_year, end_year].
    """
    print("\n--- Preprocessing: merge demand + weather ---")
    modules["build_processed_dataset"](
        countries=[country],
        years=list(range(start_year, end_year + 1)),
    )


def step_features(country: str, start_year: int, end_year: int, modules: dict):
    """
    Step 3 — Feature engineering.
    Builds lag features, calendar features and weather features
    for h+1 forecasting over [start_year, end_year].
    """
    print("\n--- Feature engineering (lags, calendar, weather) ---")
    modules["build_load_forecasting_features"](
        country=country,
        years=list(range(start_year, end_year + 1)),
        forecast_horizon=1,
    )


def step_train(country: str, start_year: int, end_year: int, modules: dict):
    """
    Step 4 — Model training.
    Trains and compares Ridge, XGBoost and LightGBM.
    Temporal splits and year range are managed by src/modeling/config.py
    and update automatically each year — start_year/end_year are ignored here.
    """
    print("\n--- Model training (Ridge / XGBoost / LightGBM) ---")
    print("    [INFO] Year range for training is managed by src/modeling/config.py")
    modules["run_training"](country=country)


def step_realtime(country: str, start_year: int, end_year: int, modules: dict):
    """
    Step 5 — Real-time ingestion (manual trigger).
    Fetches the last 48h of demand and the next 2h weather forecast.
    start_year / end_year are ignored for this step.
    For automated hourly execution, use scheduler.py instead.
    """
    meta = COUNTRIES[country]

    print("\n--- Real-time ingestion (last 48h demand + 2h weather forecast) ---")
    modules["fetch_and_store_realtime"](
        country=country,
        country_code=meta["entsoe_code"],
        latitude=meta["latitude"],
        longitude=meta["longitude"],
    )


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

STEP_FUNCTIONS = {
    "ingest":     step_ingest,
    "preprocess": step_preprocess,
    "features":   step_features,
    "train":      step_train,
    "realtime":   step_realtime,
}


def run_pipeline(steps: list, country: str, start_year: int, end_year: int):
    """
    Execute the requested pipeline steps in order.

    Parameters
    ----------
    steps : list[str]
        Ordered list of steps to run (subset of ALL_STEPS).
    country : str
        Country code to process (must be a key in COUNTRIES).
    start_year : int
        First year to process (used by ingest, preprocess, features).
    end_year : int
        Last year to process, inclusive.
    """
    # Validate inputs
    if country not in COUNTRIES:
        print(f"[ERROR] Unsupported country '{country}'. Available: {list(COUNTRIES.keys())}")
        sys.exit(1)

    if start_year > end_year:
        print(f"[ERROR] --start-year ({start_year}) must be <= --end-year ({end_year})")
        sys.exit(1)

    modules = import_modules()

    # Header
    print("=" * 60)
    print("  energy-intelligence-platform — Pipeline")
    print("=" * 60)
    print(f"  Country    : {country}")
    print(f"  Years      : {start_year} → {end_year}")
    print(f"  Steps      : {' → '.join(steps)}")
    print(f"  Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    pipeline_start = time.time()

    for i, step_name in enumerate(steps, start=1):
        print(f"\n{'=' * 60}")
        print(f"  STEP {i}/{len(steps)} : {step_name.upper()}")
        print(f"{'=' * 60}")

        step_start = time.time()

        STEP_FUNCTIONS[step_name](
            country=country,
            start_year=start_year,
            end_year=end_year,
            modules=modules,
        )

        elapsed = time.time() - step_start
        print(f"\n[OK] '{step_name}' completed in {elapsed:.1f}s")

    # Footer
    total = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"  All steps completed in {total:.1f}s")
    print(f"  Finished at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="energy-intelligence-platform — Electricity demand forecasting pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
                Examples:
                python main.py                                          # all steps, default years
                python main.py --steps ingest                          # ingestion only
                python main.py --steps preprocess features             # preprocess + features
                python main.py --steps train                           # train only
                python main.py --steps realtime                        # manual real-time cycle
                python main.py --steps ingest preprocess features \\
                                --start-year 2025 --end-year 2025       # single year
        """,
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=ALL_STEPS + ["all"],
        metavar="STEP",
        help=(
            "Steps to run (default: all). Choices:\n"
            "  ingest      → fetch raw data (ENTSO-E + Open-Meteo)\n"
            "  preprocess  → clean and merge demand + weather\n"
            "  features    → build lag / calendar / weather features\n"
            "  train       → train models and save the best one\n"
            "  realtime    → one manual real-time ingestion cycle\n"
            "  all         → run all steps in order\n"
        ),
    )

    parser.add_argument(
        "--country",
        default=DEFAULT_COUNTRY,
        choices=list(COUNTRIES.keys()),
        help=f"Country to process (default: {DEFAULT_COUNTRY})",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        dest="start_year",
        help=f"First year to process (default: {DEFAULT_START_YEAR})",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        dest="end_year",
        help=f"Last year to process, inclusive (default: current year - 1 = {DEFAULT_END_YEAR})",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    steps = ALL_STEPS if "all" in args.steps else args.steps

    run_pipeline(
        steps=steps,
        country=args.country,
        start_year=args.start_year,
        end_year=args.end_year,
    )