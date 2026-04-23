# Energy Intelligence Platform

> End-to-end machine learning system for hourly electricity demand forecasting in France: from raw API data to a live interactive dashboard.

---

## Overview

The Energy Intelligence Platform predicts France's electricity consumption one hour ahead (h+1). The system covers the full ML lifecycle: historical data ingestion, preprocessing, feature engineering, model training and comparison, real-time inference via a REST API, automated drift monitoring, and an interactive Streamlit dashboard.

Accurate short-term demand forecasting is critical for grid operators: it enables better supply/demand balancing, cost reduction, and smoother integration of renewable energy sources.

**Best model:** XGBoost, with validation MAE of **583 MW** (~1.0% of average national load).

---

## Architecture

```
ENTSO-E API ──┐
              ├──► Ingestion ──► Raw Parquet (Hive partitioned)
Open-Meteo ───┘                         │
                                        ▼
                                 Preprocessing
                              (merge, interpolation)
                                        │
                                        ▼
                              Feature Engineering
                         (lag features, calendar, weather)
                                        │
                                        ▼
                         Model Training & Comparison
                         (Ridge / XGBoost / LightGBM)
                                        │
                                        ▼
                           Best Model → best_model.pkl
                                        │
                              ┌─────────┴──────────┐
                              ▼                    ▼
                     FastAPI /predict        Streamlit Dashboard
                     (h+1 inference)         (realtime · historical · MLOps)
                              │
                              ▼
                     APScheduler (hourly)
                     + Drift Monitoring (Evidently)
```

All intermediate data is stored as partitioned Parquet files (`country=XX/year=YYYY/`) at each pipeline stage, enabling efficient querying and straightforward extension to additional countries.

---

## Project Structure

```
energy-intelligence-platform/
│
├── api/
│   └── main.py                          # FastAPI application and /predict endpoint
│
├── dashboard/
│   ├── dashboard.py                     # Entry point: page config, sidebar, routing
│   ├── tabs/
│   │   ├── realtime.py                  # Live load curve + h+1 prediction
│   │   ├── historical.py                # Dynamic year selector with auto-resampling
│   │   └── model_perf.py                # Metrics, feature importance, drift report
│   └── utils/
│       ├── data_loader.py               # Parquet/JSON reads with @st.cache_data
│       ├── prediction.py                # Feature reconstruction + model inference
│       └── charts.py                    # Reusable Plotly figures
│
├── src/
│   ├── ingestion/
│   │   ├── get_entsoe_demand.py         # Historical electricity demand (ENTSO-E)
│   │   ├── get_openmeteo_weather.py     # Historical weather (Open-Meteo)
│   │   └── get_realtime_data.py         # Rolling 192h real-time snapshot
│   ├── preprocessing/
│   │   └── build_preprocessed_dataset.py
│   ├── feature_engineering/
│   │   └── build_features.py
│   ├── modeling/
│   │   ├── config.py                    # FEATURE_COLS: single source of truth
│   │   ├── models.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── storage.py
│   └── monitoring/
│       └── monitor.py                   # Evidently drift monitoring (KS test)
│
├── data/
│   ├── raw/                             # Raw API responses
│   ├── processed/                       # Cleaned and merged data
│   ├── featured/                        # ML-ready feature datasets
│   ├── realtime/                        # Rolling real-time snapshot
│   └── monitoring/                      # Timestamped drift reports (JSON)
│
├── models/
│   ├── best_model.pkl                   # Trained XGBoost pipeline
│   └── training_results.json            # Metrics, feature names, metadata
│
├── tests/                               # 27 unit tests: all passing
│   ├── test_get_realtime_data.py
│   ├── test_build_preprocessed_dataset.py
│   ├── test_build_features.py
│   ├── test_api.py
│   └── test_monitor.py
│
├── docs/
│   └── project_assumptions_and_sources.md
│
├── notebooks/
├── main.py                              # Pipeline orchestrator (--steps flag)
├── scheduler.py                         # APScheduler: hourly real-time updates
├── Dockerfile
├── requirements.txt
└── .env                                 # API credentials (not committed, but .env.example provided)
```

---

## Prerequisites

- Python 3.12 (conda environment recommended)
- An ENTSO-E API token: free registration at [transparency.entsoe.eu](https://transparency.entsoe.eu)
- Docker (optional, for containerized deployment)

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/energy-intelligence-platform.git
cd energy-intelligence-platform
```

**2. Create and activate the conda environment**

```bash
conda create -n energy_ml_312 python=3.12
conda activate energy_ml_312
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

On macOS, XGBoost requires `libomp`:

```bash
brew install libomp
```

---

## Configuration

Create a `.env` file at the project root:

```env
ENTSOE_API_TOKEN=your_api_token_here
```

Open-Meteo does not require authentication.

---

## Usage

### Running the Pipeline

```bash
# Full pipeline: ingest → preprocess → features → train
python main.py

# Individual steps
python main.py --steps ingest
python main.py --steps preprocess features
python main.py --steps train

# Real-time data refresh + automatic drift monitoring
python main.py --steps realtime
```

**Pipeline steps:**

| Step         | Description                                                       |
| ------------ | ----------------------------------------------------------------- |
| `ingest`     | Fetch historical demand and weather from APIs (2015–2025)         |
| `preprocess` | Clean, merge, and interpolate missing values                      |
| `features`   | Build lag, calendar, and weather features                         |
| `train`      | Train Ridge, XGBoost, LightGBM; save best model                   |
| `realtime`   | Fetch the rolling 192h snapshot + run drift monitoring            |
| `monitor`    | Run drift monitoring standalone on the current real-time snapshot |

### Starting the API

```bash
# Development
python -m uvicorn api.main:app --reload

# Production
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs available at `http://localhost:8000/docs`.

### Launching the Dashboard

```bash
streamlit run dashboard/dashboard.py
```

### Scheduling Automated Updates

```bash
# Production: runs at HH:05 every hour
python scheduler.py

# Test mode: runs every N minutes
python scheduler.py --interval 2
```

---

## API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "xgboost",
  "feature_cols": ["load_t", "load_t-1", "load_t-24", ...]
}
```

### `GET /predict`

Builds features from the real-time Parquet snapshot and returns the h+1 forecast. No request body required: all feature values are reconstructed server-side from stored data.

**Response:**

```json
{
  "prediction_mw": 46234.5,
  "prediction_datetime_utc": "2025-03-29T10:00:00+00:00",
  "model": "xgboost",
  "timestamp_utc": "2025-03-29T09:00:00+00:00"
}
```

---

## Model Performance

Models are trained on French electricity data from 2015 to 2025 using a strict temporal split.

| Period     | Years     |
| ---------- | --------- |
| Training   | 2015–2023 |
| Validation | 2024      |
| Test       | 2025      |

**Results (validation MAE):**

| Model            | Val MAE (MW) |
| ---------------- | ------------ |
| Ridge Regression | —            |
| LightGBM         | —            |
| **XGBoost**      | **583**      |

XGBoost is the best model and is used for all inference.

> **Note on test set (2025):** Test MAE is significantly higher than validation MAE due to out-of-distribution conditions: exceptional cold in winter 2025 and unexpectedly low consumption in spring/autumn. This is a documented distribution shift, not a model defect. See `docs/project_assumptions_and_sources.md §4` for details.

**Input features:**

| Feature            | Description                                  |
| ------------------ | -------------------------------------------- |
| `load_t`           | Current load (MW)                            |
| `load_t-1`         | Load 1 hour ago                              |
| `load_t-24`        | Load 24 hours ago                            |
| `load_t-168`       | Load 168 hours ago (weekly seasonality)      |
| `temperature_t`    | Forecasted temperature for t+1 (aligned -1h) |
| `temperature_t-24` | Temperature 24 hours ago                     |
| `hour`             | Hour of day (0–23)                           |
| `is_weekday`       | 1 if weekday and not a public holiday        |
| `day_of_week`      | Day of week (0=Monday)                       |
| `week_of_year`     | ISO week number (1–52)                       |

Feature importance (XGBoost): `load_t` (52.8%) + `load_t-1` (38.9%) = **91.7%** of total importance.

---

## MLOps — Drift Monitoring

At each real-time ingestion cycle, the monitor compares the current 24h snapshot against the 2024 reference distribution using Evidently's KS test (threshold p < 0.05).

Eight features are monitored: the four load lags, `hour`, `day_of_week`, `is_weekday`, `week_of_year`. Temperature features are excluded, the real-time snapshot contains only ~2 hours of weather forecast, making temperature monitoring statistically meaningless.

Results are saved as timestamped JSON reports in `data/monitoring/` and surfaced in the dashboard's **Model Performance** tab.

> **Known artefact:** `hour` is structurally flagged as drifted (p ≈ 0.0) because the 24-row snapshot has a perfectly uniform hour distribution, while the 2024 reference is non-uniform. This is a false positive, documented in `docs/project_assumptions_and_sources.md §11.7`.

---

## Tests

```bash
pytest tests/ -v
```

27 tests, all passing (~2 seconds). Coverage: real-time ingestion, preprocessing, feature engineering, API endpoints, and drift monitoring.

---

## Docker

```bash
# Build
docker build -t energy-intelligence-platform .

# Run (mount data volume, pass ENTSO-E token)
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  energy-intelligence-platform
```

---

## Data Sources

**ENTSO-E Transparency Platform**

- URL: [transparency.entsoe.eu](https://transparency.entsoe.eu)
- Data: Actual hourly electricity demand for France (Document type A65, Process type A16)
- Authentication: API token required (free registration)

**Open-Meteo**

- URL: [open-meteo.com](https://open-meteo.com)
- Data: Hourly temperature, humidity, wind speed, solar radiation
- Authentication: None required
- Endpoints: Archive API (historical) and Forecast API (real-time)

---

## Documentation

Design decisions, data assumptions, known limitations, and source references are documented in [`docs/project_assumptions_and_sources.md`](docs/project_assumptions_and_sources.md).
