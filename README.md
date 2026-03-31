# Energy Demand Intelligence Platform

A production-grade machine learning system for forecasting hourly electricity demand in France. The platform ingests real-time and historical data from ENTSO-E and Open-Meteo, trains and compares multiple ML models, and serves predictions through a REST API.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Pipeline](#running-the-pipeline)
  - [Starting the API](#starting-the-api)
  - [Scheduling Automated Updates](#scheduling-automated-updates)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)
- [Docker](#docker)
- [Data Sources](#data-sources)
- [Notebooks](#notebooks)

---

## Overview

The Energy Demand Intelligence Platform predicts France's electricity load one hour ahead (h+1). Accurate demand forecasting is critical for grid operators to balance supply, reduce costs, and integrate renewable energy sources.

**Key capabilities:**

- Historical data ingestion from 2015 to 2024 (ENTSO-E + Open-Meteo)
- Automated hourly real-time data updates via a built-in scheduler
- Feature engineering with lag features, calendar features, and weather signals
- Multi-model training and comparison (Ridge, XGBoost, LightGBM)
- FastAPI REST endpoint for on-demand predictions
- Docker support for production deployment

---

## Architecture

```
ENTSO-E API ──┐
              ├──► Ingestion ──► Raw Data (Parquet)
Open-Meteo ───┘                        │
                                        ▼
                                 Preprocessing
                                        │
                                        ▼
                              Feature Engineering
                                        │
                                        ▼
                               Model Training
                          (Ridge / XGBoost / LightGBM)
                                        │
                                        ▼
                              Best Model Saved
                                        │
                                        ▼
                              FastAPI Prediction API
```

Data is stored as partitioned Parquet files (`country=XX/year=YYYY/`) at each stage, enabling efficient querying and straightforward extension to additional countries.

---

## Project Structure

```
energy-intelligence-platform/
│
├── api/
│   └── app.py                  # FastAPI application and prediction endpoints
│
├── src/
│   ├── ingestion/
│   │   ├── get_entsoe_demand.py         # Historical electricity demand from ENTSO-E
│   │   ├── get_openmeteo_weather.py     # Historical weather from Open-Meteo
│   │   └── get_realtime_data.py         # Real-time data fetching (last 48h)
│   │
│   ├── preprocessing/
│   │   └── build_preprocessed_dataset.py  # Cleaning, merging, interpolation
│   │
│   ├── feature_engineering/
│   │   └── build_features.py            # Lag, calendar, and weather features
│   │
│   ├── modeling/
│   │   └── train.py                     # Model training, evaluation, and export
│   │
│   └── evaluation/
│       └── metrics.py                   # MAE and RMSE utilities
│
├── data/
│   ├── raw/                    # Raw API responses (partitioned Parquet)
│   ├── processed/              # Cleaned and merged data
│   ├── featured/               # ML-ready feature datasets
│   └── realtime/               # Rolling 7-day real-time data
│
├── models/
│   ├── best_model.pkl          # Trained XGBoost pipeline (sklearn)
│   └── training_results.json   # Model metadata and performance metrics
│
├── notebooks/                  # Exploratory and step-by-step notebooks
│
├── main.py                     # Pipeline orchestrator
├── scheduler.py                # Automated hourly ingestion scheduler
├── Dockerfile
├── requirements.txt
└── .env                        # API credentials (not committed)
```

---

## Prerequisites

- Python 3.7+
- An ENTSO-E API token (free registration at [transparency.entsoe.eu](https://transparency.entsoe.eu))
- Docker (optional, for containerized deployment)

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/energy-intelligence-platform.git
cd energy-intelligence-platform
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file at the root of the project:

```env
ENTSOE_API_TOKEN=your_api_token_here
```

Your ENTSO-E token is the only required credential. Open-Meteo does not require authentication.

---

## Usage

### Running the Pipeline

The `main.py` orchestrator runs the full pipeline or individual steps.

```bash
# Run the complete pipeline (ingest → preprocess → features → train)
python main.py

# Limit to a specific year range
python main.py --start-year 2022 --end-year 2024

# Run individual steps
python main.py --steps ingest
python main.py --steps preprocess features
python main.py --steps train

# Trigger a real-time data refresh
python main.py --steps realtime
```

**Pipeline steps:**

| Step         | Description                                                     |
| ------------ | --------------------------------------------------------------- |
| `ingest`     | Fetch historical demand and weather data from APIs              |
| `preprocess` | Clean, merge, and interpolate missing values                    |
| `features`   | Build lag, calendar, and weather features                       |
| `train`      | Train Ridge, XGBoost, and LightGBM; save best model             |
| `realtime`   | Fetch the last 48 hours of demand and a 2-hour weather forecast |

### Starting the API

```bash
# Development (auto-reload on code changes)
uvicorn api.app:app --reload

# Production
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive documentation is at `http://localhost:8000/docs`.

### Scheduling Automated Updates

The scheduler runs real-time ingestion automatically.

```bash
# Production mode: runs hourly at HH:05
python scheduler.py

# Test mode: runs every N minutes
python scheduler.py --interval 2

# Specify a country explicitly
python scheduler.py --country FR
```

---

## API Reference

### `GET /`

Returns API status and a link to the documentation.

### `GET /health`

Returns model loading status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "xgboost"
}
```

### `GET /model/info`

Returns model metadata, feature names, training/validation/test metrics, and date ranges.

### `POST /predict`

Returns an h+1 electricity load forecast.

**Request body:**

```json
{
  "load_t": 45000.0,
  "load_t_minus_1": 44800.0,
  "load_t_minus_24": 43500.0,
  "load_t_minus_168": 46200.0,
  "temperature_t": 12.5,
  "hour": 9,
  "is_weekday": 1,
  "week_of_year": 12
}
```

| Field              | Type  | Description                                          |
| ------------------ | ----- | ---------------------------------------------------- |
| `load_t`           | float | Current electricity load in MW                       |
| `load_t_minus_1`   | float | Load 1 hour ago in MW                                |
| `load_t_minus_24`  | float | Load 24 hours ago in MW                              |
| `load_t_minus_168` | float | Load 168 hours (1 week) ago in MW                    |
| `temperature_t`    | float | Current temperature in °C                            |
| `hour`             | int   | Current hour (0–23)                                  |
| `is_weekday`       | int   | 1 if weekday (and not a public holiday), 0 otherwise |
| `week_of_year`     | int   | ISO week number (1–52)                               |

**Response:**

```json
{
  "prediction_mw": 46234.5,
  "prediction_datetime_utc": "2024-03-29T10:00:00+00:00",
  "model": "xgboost",
  "timestamp_utc": "2024-03-29T09:00:00+00:00"
}
```

---

## Model Performance

Models are trained on French electricity data from 2015 to 2024 using a strict temporal split to prevent data leakage.

| Period     | Years     |
| ---------- | --------- |
| Training   | 2015–2022 |
| Validation | 2023      |
| Test       | 2024      |

**Results:**

| Model            | Validation MAE (MW) | Test MAE (MW) |
| ---------------- | ------------------- | ------------- |
| Ridge Regression | —                   | —             |
| LightGBM         | —                   | —             |
| **XGBoost**      | **701.71**          | **594.78**    |

XGBoost was selected as the best model and is used for all API predictions.

**Input features:**

| Feature            | Description                      |
| ------------------ | -------------------------------- |
| `load_t`           | Current load                     |
| `load_t_minus_1`   | 1-hour lag                       |
| `load_t_minus_24`  | 24-hour lag                      |
| `load_t_minus_168` | Weekly lag                       |
| `temperature_t`    | Current temperature              |
| `hour`             | Hour of day                      |
| `is_weekday`       | Weekday / weekend / holiday flag |
| `week_of_year`     | Seasonal signal                  |

---

## Docker

```bash
# Build the image
docker build -t energy-intelligence-platform .

# Run the container
docker run -p 8000:8000 energy-intelligence-platform
```

The API will be accessible at `http://localhost:8000`.

---

## Data Sources

### ENTSO-E Transparency Platform

- **URL**: [transparency.entsoe.eu](https://transparency.entsoe.eu)
- **Data**: Actual hourly electricity demand for France
- **Document type**: A65 (Actual Total Load), Process type: A16 (Realised)
- **Authentication**: API token required (free registration)

### Open-Meteo

- **URL**: [open-meteo.com](https://open-meteo.com)
- **Data**: Hourly weather — temperature, humidity, wind speed, solar radiation
- **Authentication**: None required
- **Endpoints**: Archive API (historical) and Forecast API (real-time)

---

## Notebooks

Step-by-step Jupyter notebooks walk through several stages of the pipeline.
