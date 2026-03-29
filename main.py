"""
energy-intelligence-platform — Pipeline Orchestrator
=====================================================

Single entry point to run all or part of the data pipeline.

Usage:
------
# Run all steps (ingest → preprocess → features → train)
    python main.py

# Run all steps over a specific year range
    python main.py --start-year 2023 --end-year 2024

# Run ingestion only
    python main.py --steps ingest

# Run preprocessing and feature engineering only
    python main.py --steps preprocess features

# Train models only
    python main.py --steps train

# Run one real-time ingestion cycle manually
    python main.py --steps realtime

# Display help
    python main.py --help
"""

import argparse
import sys
import time
from datetime import datetime


# ---------------------------------------------------------------------
# Centralized configuration
# ---------------------------------------------------------------------

COUNTRIES = {
    "FR": {
        "entsoe_code": "10YFR-RTE------C",
        "latitude": 48.8534,
        "longitude": 2.3488,
    }
}

# Steps that require --start-year / --end-year
YEAR_DEPENDENT_STEPS = ["ingest", "preprocess", "features"]

# All available steps
ALL_STEPS = ["ingest", "preprocess", "features", "train", "realtime"]

DEFAULT_COUNTRY = "FR"
DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2024


# ---------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------

def import_modules():
    """
    Import project modules.
    Imports are done here (rather than at the top of the file) to display
    a clear error message if a module is missing.
    """
    try:
        from src.ingestion.get_entsoe_demand import fetch_entsoe_demand_and_store
        from src.ingestion.get_openmeteo_weather import fetch_openmeteo_weather_and_store
        from src.ingestion.get_realtime_data import fetch_and_store_realtime
        from src.preprocessing.build_preprocessed_dataset import build_processed_dataset
        from src.feature_engineering.build_features import build_load_forecasting_features
        from src.modeling.train import run_training
        return {
            "fetch_entsoe_demand_and_store": fetch_entsoe_demand_and_store,
            "fetch_openmeteo_weather_and_store": fetch_openmeteo_weather_and_store,
            "fetch_and_store_realtime": fetch_and_store_realtime,
            "build_processed_dataset": build_processed_dataset,
            "build_load_forecasting_features": build_load_forecasting_features,
            "run_training": run_training,
        }
    except ImportError as e:
        print(f"\n[ERROR] Failed to import a module: {e}")
        print("-> Make sure you run main.py from the project root.")
        print("-> Example: python main.py --steps all\n")
        sys.exit(1)


# ---------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------

def step_ingest(country: str, start_year: int, end_year: int, modules: dict):
    """Step 1: Raw data ingestion (ENTSO-E + Open-Meteo)."""

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
    """Step 2: Preprocessing and merging of raw data."""

    years = list(range(start_year, end_year + 1))

    print("\n--- Preprocessing and merging demand + weather ---")
    modules["build_processed_dataset"](
        countries=[country],
        years=years,
    )


def step_features(country: str, start_year: int, end_year: int, modules: dict):
    """Step 3: Feature engineering for h+1 forecasting."""

    years = list(range(start_year, end_year + 1))

    print("\n--- Feature engineering (lag features, calendar, weather) ---")
    modules["build_load_forecasting_features"](
        country=country,
        years=years,
        forecast_horizon=1,
    )


def step_train(country: str, start_year: int, end_year: int, modules: dict):
    """Step 4: Train and compare models, save the best one."""

    years = list(range(start_year, end_year + 1))

    print("\n--- Model training (Ridge vs XGBoost vs LightGBM) ---")
    modules["run_training"](
        country=country,
        years=years,
    )


def step_realtime(country: str, start_year: int, end_year: int, modules: dict):
    """
    Step 5: One real-time ingestion cycle (fetch latest data + store).

    Note: start_year / end_year are ignored for this step.
    For automated hourly execution, use scheduler.py instead.
    """

    meta = COUNTRIES[country]

    print("\n--- Real-time ingestion (last 48h demand + weather forecast) ---")
    modules["fetch_and_store_realtime"](
        country=country,
        country_code=meta["entsoe_code"],
        latitude=meta["latitude"],
        longitude=meta["longitude"],
    )


# ---------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------

def run_pipeline(steps: list, country: str, start_year: int, end_year: int):
    """
    Execute pipeline steps in order.

    Parameters
    ----------
    steps : list[str]
        List of steps to run
    country : str
        Country code (e.g. "FR")
    start_year : int
        First year to process (used by ingest, preprocess, features, train)
    end_year : int
        Last year to process inclusive (same as above)
    """

    # Parameter validation
    if country not in COUNTRIES:
        print(f"[ERROR] Country '{country}' is not supported. Available: {list(COUNTRIES.keys())}")
        sys.exit(1)

    if start_year > end_year:
        print(f"[ERROR] start_year ({start_year}) must be <= end_year ({end_year})")
        sys.exit(1)

    # Import modules
    modules = import_modules()

    # Pipeline summary
    print("=" * 60)
    print("  energy-intelligence-platform -- Pipeline")
    print("=" * 60)
    print(f"  Country    : {country}")
    print(f"  Years      : {start_year} -> {end_year}")
    print(f"  Steps      : {' -> '.join(steps)}")
    print(f"  Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    pipeline_start = time.time()

    # Map step names to their functions
    step_functions = {
        "ingest":     step_ingest,
        "preprocess": step_preprocess,
        "features":   step_features,
        "train":      step_train,
        "realtime":   step_realtime,
    }

    for i, step_name in enumerate(steps, start=1):
        print(f"\n{'=' * 60}")
        print(f"  STEP {i}/{len(steps)} : {step_name.upper()}")
        print(f"{'=' * 60}")

        step_start = time.time()

        step_functions[step_name](
            country=country,
            start_year=start_year,
            end_year=end_year,
            modules=modules,
        )

        step_duration = time.time() - step_start
        print(f"\n[OK] Step '{step_name}' completed in {step_duration:.1f}s")

    # Final summary
    total_duration = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline completed in {total_duration:.1f}s")
    print(f"  Finished at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


# ---------------------------------------------------------------------
# Command line interface (CLI)
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="energy-intelligence-platform -- Electricity demand forecasting pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --start-year 2023 --end-year 2024
  python main.py --steps ingest
  python main.py --steps preprocess features
  python main.py --steps train
  python main.py --steps realtime
  python main.py --steps all --country FR --start-year 2015 --end-year 2024
        """
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=ALL_STEPS + ["all"],
        metavar="STEP",
        help=(
            "Steps to run. Possible values:\n"
            "  ingest      -> Fetch raw data (ENTSO-E + Open-Meteo)\n"
            "  preprocess  -> Clean and merge demand + weather\n"
            "  features    -> Feature engineering for h+1 forecasting\n"
            "  train       -> Train and compare models, save best\n"
            "  realtime    -> One real-time ingestion cycle (manual trigger)\n"
            "  all         -> Run all steps (default)\n"
        )
    )

    parser.add_argument(
        "--country",
        default=DEFAULT_COUNTRY,
        choices=list(COUNTRIES.keys()),
        help=f"Country code to process (default: {DEFAULT_COUNTRY})"
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        dest="start_year",
        help=f"First year to process (default: {DEFAULT_START_YEAR})"
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        dest="end_year",
        help=f"Last year to process, inclusive (default: {DEFAULT_END_YEAR})"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # If "all" is in steps, replace with the full ordered list
    steps = ALL_STEPS if "all" in args.steps else args.steps

    run_pipeline(
        steps=steps,
        country=args.country,
        start_year=args.start_year,
        end_year=args.end_year,
    )