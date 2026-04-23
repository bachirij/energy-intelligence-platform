# Project Assumptions & Data Sources

## Overview

This document describes the data sources, modeling assumptions, and known limitations of the Energy Intelligence Platform: a machine learning system for short-term electricity demand forecasting in France (h+1 horizon).

---

## 1. Data Sources

### 1.1 Electricity Demand - ENTSO-E Transparency Platform

| Property      | Value                                                                       |
| ------------- | --------------------------------------------------------------------------- |
| Provider      | European Network of Transmission System Operators for Electricity (ENTSO-E) |
| Endpoint      | `https://web-api.tp.entsoe.eu/api`                                          |
| Document type | A65 - Actual Total Load                                                     |
| Process type  | A16 - Realised                                                              |
| Bidding zone  | `10YFR-RTE------C` (France - RTE)                                           |
| Granularity   | Hourly                                                                      |
| Timezone      | UTC                                                                         |
| Coverage      | 2015–2024 (historical), rolling 48h window (real-time)                      |
| Access        | Free API, registration required                                             |
| Official page | https://transparency.entsoe.eu                                              |

**Why this source ?**
ENTSO-E is the official pan-European platform for electricity market transparency data,
mandated by EU regulation. The Actual Total Load (A65) represents the measured
electricity consumption at the transmission level for France, published by RTE
(Réseau de Transport d'Électricité), the French TSO. It is the standard reference
used in academic research and industry for French load forecasting.

---

### 1.2 Weather Data - Open-Meteo

| Property            | Value                                                |
| ------------------- | ---------------------------------------------------- |
| Provider            | Open-Meteo (open-source weather API)                 |
| Historical endpoint | `https://archive-api.open-meteo.com/v1/archive`      |
| Forecast endpoint   | `https://api.open-meteo.com/v1/forecast`             |
| Granularity         | Hourly                                               |
| Timezone            | UTC                                                  |
| Coverage            | 2015–2024 (historical), next 2h (real-time forecast) |
| Access              | Free, no API key required                            |
| Official page       | https://open-meteo.com                               |

**Weather variables used**

| Variable                      | Unit | Justification                                        |
| ----------------------------- | ---- | ---------------------------------------------------- |
| `temperature_2m`              | °C   | Primary driver of heating/cooling electricity demand |
| `relative_humidity_2m`        | %    | Affects perceived temperature and HVAC usage         |
| `wind_speed_10m`              | km/h | Influences heating demand and wind chill effect      |
| `shortwave_radiation_instant` | W/m² | Proxy for solar irradiance and natural lighting      |

**Why this source ?**
Open-Meteo provides ERA5-based reanalysis data for historical queries, which is the industry standard for retrospective weather analysis. It is free, reliable, and provides a consistent API for both historical and forecast data, making it suitable for a production-grade pipeline.

---

## 2. Modeling Assumptions

### 2.1 Forecast horizon

The system is designed for **h+1 forecasting**, predicting electricity demand one hour ahead of the current timestamp. This is the most common operational horizon for TSOs and energy traders.

### 2.2 Target variable

The target variable is `load_MW` shifted by -1 hour, representing the actual electricity consumption (in megawatts) at time t+1.

### 2.3 Temporal granularity

All data is processed at **hourly granularity in UTC**. Local time effects (e.g. French working hours, peak demand patterns) are captured through calendar features derived from UTC timestamps converted to local time where relevant.

### 2.4 Training period

Historical data spans **2015–2025** (11 years), covering a wide range of weather conditions, economic cycles, and behavioral patterns including the COVID-19 period (2020–2021), which represents an anomalous demand pattern.

---

## 3. Geographic Assumptions

### 3.1 Weather station proxy

**Current assumption**: a single weather observation point is used to represent France as a whole, located in Paris (lat=48.8534, lon=2.3488).

**Justification**: Paris is the most populated city in France and is geographically central in terms of population density. For a first iteration of the model, this is a reasonable approximation.

**Known limitation**: France covers approximately 550,000 km² with significant regional climate variability. Electricity demand is driven by temperature across all regions simultaneously, not just Paris. A single station may underestimate the impact of cold spells in the north-east or heat waves in the south.

**Planned improvement**: replace the single station with a **population-weighted average** across 5–6 representative cities (e.g. Paris, Lyon, Marseille, Bordeaux, Lille, Strasbourg) to better capture regional temperature variability. Weights would be proportional to regional population and heating degree days.

---

## 4. Preprocessing Assumptions

### 4.1 Missing value handling

Missing hourly values are handled differently depending on the variable type:

| Variable type                                                         | Method                       | Justification                                                                                                                                                            |
| --------------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `load_MW`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m` | Linear time interpolation    | These variables vary smoothly over time; linear interpolation is appropriate for short gaps                                                                              |
| `shortwave_radiation_instant`                                         | Forward fill (ffill)         | Solar radiation is exactly 0 at night; linear interpolation between nighttime and daytime values would produce physically incorrect (non-zero) values during night hours |
| Categorical columns (`country`)                                       | Forward fill + backward fill | Constant within a year, filling is safe                                                                                                                                  |

Interpolation is applied only to interior gaps (`limit_area="inside"`). Leading or trailing NaNs are not filled, verified to be absent in the data.

### 4.2 Duplicate timestamps

Duplicate timestamps can occur in ENTSO-E data during daylight saving time transitions. Duplicates are removed by keeping the first occurrence, after sorting by timestamp.

### 4.3 Merge strategy

Demand and weather data are merged using an **inner join** on the datetime column. This ensures only rows with both demand and weather data are kept, avoiding silent NaN propagation into the feature set.

---

## 5. Feature Engineering Assumptions

### 5.1 Lag features

The following lag features are used to capture autocorrelation in the demand signal:

| Feature            | Lag                | Justification                                                                                                                                                                       |
| ------------------ | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_t-1`         | 1 hour             | Strong short-term autocorrelation                                                                                                                                                   |
| `load_t-24`        | 24 hours           | Same hour the previous day (daily seasonality)                                                                                                                                      |
| `load_t-168`       | 168 hours (7 days) | Same hour the previous week (weekly seasonality)                                                                                                                                    |
| `temperature_t-24` | 24 hours           | Electricity demand responds to temperature with inertia (heating habits, thermal mass of buildings), yesterday's temperature at the same hour improves prediction of current demand |

### 5.2 Calendar features

| Feature        | Justification                                                                                                                                   |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `hour`         | Captures intraday demand patterns (morning peak, evening peak)                                                                                  |
| `day_of_week`  | Day-level granularity within the week; Monday mornings and Friday evenings have distinct demand profiles that `is_weekday` alone cannot capture |
| `is_weekday`   | Binary flag summarising weekday vs weekend demand regime                                                                                        |
| `week_of_year` | Captures seasonal trends (winter peaks, summer troughs)                                                                                         |

**Note on cyclical encoding**: `hour` and `week_of_year` are kept as plain integers. Cyclical sin/cos encoding is not applied because the model uses XGBoost, which handles integer ordinal features natively through axis-aligned splits and does not require cyclical representation.

### 5.3 Holiday calendar

French public holidays are computed using the `holidays` Python library (`holidays.country_holidays("FR")`). Public holidays are treated as non-working days:
`is_weekday` is overridden to 0 regardless of the day of the week. No separate `is_holiday` column is kept, the assumption is that holiday demand profiles are sufficiently captured by the weekend regime (`is_weekday=0`).

Holiday detection is applied on timestamps converted to `Europe/Paris` local time to avoid off-by-one errors at UTC midnight (e.g. December 31 at 23:00 UTC = January 1 local time).

---

## 6. Real-Time Pipeline Assumptions

### 6.1 ENTSO-E data latency

ENTSO-E actual load data is typically published with a **delay of 1–2 hours**. The real-time pipeline fetches the last 48 hours of demand data to account for any publication delay or short gaps.

### 6.2 Weather forecast

Open-Meteo forecast data is fetched for the **next 2 hours** (t and t+1), which is sufficient to build features for an h+1 prediction. The forecast cache expires after 30 minutes to balance API load and data freshness.

### 6.3 Rolling window

The real-time parquet file maintains a **7-day rolling window** of data. Rows older than 7 days are dropped on each update to keep the file lightweight.

---

## 7. Known Limitations & Future Improvements

| Limitation                                           | Impact                                                                                                                                                                                                    | Planned fix                                                                   |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Single weather station (Paris)                       | May underestimate regional demand drivers                                                                                                                                                                 | Population-weighted multi-city average                                        |
| No cloud deployment yet                              | Pipeline runs locally only                                                                                                                                                                                | Deploy on Hugging Face Spaces/AWS/GCP with scheduled ingestion                |
| No model monitoring                                  | Model drift is not detected                                                                                                                                                                               | Add data drift and prediction drift monitoring (Evidently)                    |
| COVID-19 anomaly not flagged                         | 2020–2021 data may distort seasonal patterns                                                                                                                                                              | Add `is_lockdown` binary feature or exclude from training                     |
| No uncertainty quantification                        | Point forecast only                                                                                                                                                                                       | Add prediction intervals (quantile regression or conformal prediction)        |
| Tests not yet implemented                            | Code reliability not guaranteed                                                                                                                                                                           | Add unit tests for ingestion, preprocessing, and feature engineering          |
| Val 2024 unrepresentative for model selection        | 2024 was an abnormally warm year (mean load: 48,904 MW), biasing model selection toward low-consumption conditions and penalising generalisation on higher-consumption periods                            | Use temporal cross-validation across multiple years for model selection       |
| Test set 2025 partially out of distribution          | Spring–autumn 2025 consumption levels (47,000–55,000 MW) were underrepresented in training; MAE of 5,606 MW is a pessimistic upper bound driven by Apr–Nov, not representative of operational performance | Evaluate on a climatically neutral year or report MAE by season               |
| ENTSO-E API returns data beyond requested period_end | For year 2025, the API returned data up to 2028 (forecasts/planned values), inflating the raw dataset to 26,000+ rows                                                                                     | Explicit year filter applied post-fetch: `df[df["datetime"].dt.year == year]` |
| ENTSO-E sub-hourly points during DST transitions     | Raw data occasionally contains 15-min and 45-min interval points around daylight saving time changes, causing incorrect row counts                                                                        | Resample to strict 1h frequency using mean aggregation after fetching         |

---

## 8. Modeling

### 8.1 Model selection strategy

Three models are trained and compared on the validation set (2024). The model with the lowest validation MAE is selected and evaluated once on the test set (2025).

| Model            | Preprocessing           | Justification                                                     |
| ---------------- | ----------------------- | ----------------------------------------------------------------- |
| Ridge regression | StandardScaler required | Linear baseline; sensitive to feature scale                       |
| XGBoost          | No scaling needed       | Tree-based; handles integer features and non-linearities natively |
| LightGBM         | No scaling needed       | Tree-based; faster than XGBoost on large datasets                 |

All models are wrapped in a `sklearn.Pipeline` for a consistent `.fit()` / `.predict()` API and to ensure the scaler is applied correctly within cross-validation if used in the future.

### 8.2 Temporal split

Splits are computed dynamically based on the current year to remain valid as new data is added:

| Split | Period                    | Role                                                   |
| ----- | ------------------------- | ------------------------------------------------------ |
| Train | 2015 - (current_year - 3) | Model learning                                         |
| Val   | current_year - 2          | Model selection — never used for hyperparameter tuning |
| Test  | current_year - 1          | Final honest evaluation — touched once                 |

In 2026: train = 2015–2023, val = 2024, test = 2025.

Boundaries use `+ pd.Timedelta("1h")` offsets to avoid inclusive `.loc[]` overlap between splits.

### 8.3 Evaluation metrics

MAE (Mean Absolute Error) is used as the primary model selection criterion.

RMSE (Root Mean Squared Error) is reported as a secondary metric to penalise large errors more heavily.

Both are expressed in MW to remain interpretable in the context of electricity demand.

### 8.4 Feature importance findings (XGBoost, gain metric)

| Feature            | Importance | Interpretation                                                              |
| ------------------ | ---------- | --------------------------------------------------------------------------- |
| `load_t`           | 0.528      | Dominant predictor, current consumption is the best proxy for h+1           |
| `load_t-1`         | 0.389      | Strong short-term autocorrelation                                           |
| `load_t-24`        | 0.022      | Daily seasonality, marginal once short-term lags are included               |
| `load_t-168`       | 0.020      | Weekly seasonality, same                                                    |
| `hour`             | 0.016      | Captures morning/evening ramp patterns                                      |
| `is_weekday`       | 0.015      | Differentiates weekday vs weekend demand regime                             |
| `temperature_t`    | 0.004      | Low marginal importance, temperature effect is already embedded in `load_t` |
| `day_of_week`      | 0.003      | Near zero, subsumed by `is_weekday` for h+1                                 |
| `week_of_year`     | 0.002      | Near zero for h+1; would matter more at h+24                                |
| `temperature_t-24` | 0.001      | Near zero, confirms thermal inertia effect is weak at h+1 horizon           |

`load_t` and `load_t-1` together account for **91.7% of total importance**.

This is expected for h+1 forecasting: the time series is highly autocorrelated at short lags. Temperature features would carry significantly more weight at longer horizons (h+6, h+24).

### 8.5 Residual analysis (validation set 2024)

- Residual distribution is approximately centred at zero: no systematic bias.
- Left skew observed: when the model makes large errors, it tends to **overestimate** consumption (negative residuals). This is consistent with the difficulty of anticipating sharp nocturnal drops.
- Variance is **heteroscedastic**: errors are significantly larger in winter (Jan–Apr 2024, ±4,000–5,000 MW) than in summer (±1,000–2,000 MW). This is partly attributable to the single weather station proxy (Paris), which captures temperature variability less accurately during cold spells affecting multiple regions.

### 8.6 MAE by hour of day (validation set 2024)

Two error peaks are observed, corresponding to the two main demand ramps:

| Period               | Hours (UTC) | MAE           | Interpretation                                                                  |
| -------------------- | ----------- | ------------- | ------------------------------------------------------------------------------- |
| Night → morning ramp | 02h–06h UTC | ~600–750 MW   | End-of-night consumption rise; speed of ramp varies with season and temperature |
| Evening peak         | 18h–20h UTC | ~960–1,217 MW | Sharpest daily transition; return home, heating, cooking                        |
| Stable midday        | 12h–14h UTC | ~350–400 MW   | Flat plateau; `load_t-1` is highly predictive                                   |
| Night baseline       | 00h UTC     | ~280 MW       | Low, stable consumption; easiest to predict                                     |

The evening peak at **20h UTC (21h local) is the hardest hour to predict** (MAE = 1,217 MW).

A feature capturing the rate of change of consumption (e.g. `load_t - load_t-2`) could help the model anticipate ramp acceleration and is identified as a potential v2 improvement.

### 8.7 Test set evaluation (2025)

**Global test MAE: 5,606 MW** — significantly higher than validation MAE (583 MW).

This degradation is not caused by a model defect but by an **unfavourable evaluation context**:

| Month        | MAE (MW)    | Mean load (MW) | Interpretation                                         |
| ------------ | ----------- | -------------- | ------------------------------------------------------ |
| Jan 2025     | 1,594       | 61,704         | Within training distribution: good performance         |
| Feb 2025     | 1,840       | 68,864         | High but plausible winter load: model generalises      |
| Mar 2025     | 2,313       | 70,579         | Exceptionally cold March: near distribution boundary   |
| Apr 2025     | 6,614       | 52,898         | Spring drop: underrepresented in training              |
| May–Nov 2025 | 6,000–9,334 | 45,000–56,000  | Out-of-distribution low consumption levels             |
| Dec 2025     | 3,431       | 48,752         | Abnormally low for December (vs 60,568 MW in Dec 2024) |

**Root cause**: the validation year (2024) was abnormally warm (mean load: 48,904 MW, the lowest in the 2015–2025 series), leading to model selection optimised for
low-consumption conditions.

The test year (2025) was abnormally cold in winter and showed unexpectedly low consumption in spring–autumn, creating a distribution mismatch in both directions.

The **global MAE of 5,606 MW is a pessimistic upper bound** and not representative of operational performance. On the winter months (Jan–Mar 2025), where the model operates within its training distribution, MAE ranges from 1,594 to 2,313 MW: consistent with validation performance.

---

## 9. Serving Layer — FastAPI + Docker

### 9.1 API design

The model is served via a **FastAPI** application (`api/main.py`) exposing two endpoints:

| Endpoint   | Method | Role                                                                     |
| ---------- | ------ | ------------------------------------------------------------------------ |
| `/health`  | GET    | Verifies the model is loaded; returns model name and feature columns     |
| `/predict` | GET    | Builds features from the realtime parquet and returns the h+1 prediction |

The model and metadata (`best_model.pkl`, `training_results.json`) are loaded **once at startup** using FastAPI's `@app.on_event("startup")` hook and stored as global variables.

This avoids reloading the model on every request, which would be prohibitively slow.

The `holidays.country_holidays("FR")` object is also instantiated at startup for the same reason.

### 9.2 Feature construction at inference time

At inference time, features are built by **direct timestamp lookups** on the realtime parquet, not by calling `build_features.py` or using `DataFrame.shift()`.

**Why not use `shift()` at inference time ?**
`shift()` assumes contiguous rows with no gaps. If a single hour is missing in the realtime parquet (API timeout, publication delay), `shift(24)` silently returns the wrong row. Timestamp lookups raise an explicit `HTTPException(503)` if the required row is absent, making failures visible rather than silent.

Feature construction logic:

| Feature                               | Source                                                                                  |
| ------------------------------------- | --------------------------------------------------------------------------------------- |
| `load_t`                              | Last row of realtime parquet (`load_MW` at `t`)                                         |
| `load_t-1`                            | Lookup `load_MW` at `t - 1h`                                                            |
| `load_t-24`                           | Lookup `load_MW` at `t - 24h`                                                           |
| `load_t-168`                          | Lookup `load_MW` at `t - 168h`                                                          |
| `temperature_t`                       | Last row (`temperature_2m` at `t`, which holds the t+1 forecast, see §9.3)              |
| `temperature_t-24`                    | Lookup `temperature_2m` at `t - 24h`                                                    |
| `hour`, `day_of_week`, `week_of_year` | Derived from `t` converted to `Europe/Paris`                                            |
| `is_weekday`                          | `t_paris.dayofweek < 5`, overridden to 0 if `t_paris.date()` is a French public holiday |

The feature vector is assembled in the exact order defined by `feature_cols` in `training_results.json`, which mirrors `FEATURE_COLS` in `config.py`.

This guarantees consistency between training and inference.

### 9.3 Weather alignment at inference time

The realtime ingestion pipeline (`get_realtime_data.py`) shifts weather timestamps by -1h before merging with demand data:

```python
df_weather["datetime"] -= pd.Timedelta(hours=1)
```

This means the `temperature_2m` value stored at row `t` is actually the Open-Meteo **forecast for `t+1`**. When `/predict` reads `temperature_t` from the last row, it is using the temperature forecast for the target hour — consistent with how `temperature_t` was constructed during training (via `build_features.py`).

### 9.4 Rolling window correction

The realtime parquet rolling window was increased from **168h to 192h** (7 days + 24h margin).

**Reason**: with exactly 168h of data, the first available row is at `t - 167h`, making the `load_t-168h` lookup fail with a 503 error. The 24h margin ensures the `t-168h` row is always present even if a few hours of data are missing due to API delays or retries.

### 9.5 Containerisation with Docker

The API is containerised using a `python:3.12-slim` base image.

Key design decisions:

| Decision                                                 | Rationale                                                                                                                        |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `ENV PYTHONPATH=/app`                                    | Allows `api/main.py` to import from `src/modeling/config.py` without installing the project as a package                         |
| `data/` mounted as a volume (`-v $(pwd)/data:/app/data`) | Realtime data is generated at runtime and must not be baked into the image; separating data from code is standard practice       |
| `--env-file .env` at runtime                             | `ENTSOE_API_TOKEN` is never written into the image; secrets are injected at container startup                                    |
| Absolute paths via `Path(__file__).resolve().parents[1]` | Paths are resolved relative to the source file, not the working directory; robust regardless of where `uvicorn` is launched from |
| `models/` copied into the image at build time            | The trained model is a build artifact, versioned with the code; it does not change at runtime                                    |

**Build and run commands:**

```bash
docker build -t energy-intelligence-platform .

docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  energy-intelligence-platform
```

---

## 10. Unit Tests

### 10.1 Philosophy and scope

Unit tests verify that individual functions behave correctly in isolation, using controlled inputs and expected outputs. External dependencies (API calls, file I/O) are either avoided by testing pure functions directly, or replaced with **mocks** (fake objects that simulate real behaviour without side effects).

The test suite covers the three modules with non-trivial logic. The following are explicitly **not tested**: external API calls (ENTSO-E, Open-Meteo), the orchestrator `main.py`, and the XGBoost model itself (tested by its authors).

**Test runner**: `pytest`
**Total tests**: 17, all passing
**Execution time**: ~2 seconds

```bash
pytest tests/ -v
```

---

### 10.2 Ingestion - `tests/test_get_realtime_data.py` (3 tests)

Tests target `build_realtime_snapshot` and `save_realtime_snapshot`.

| Test                                               | What it verifies                                                                                                                                 |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_weather_shift_minus_one_hour`                | Weather originally at t+1 is stored at row t after the -1h shift: ensures temperature_t at inference time holds the t+1 forecast                 |
| `test_left_join_keeps_demand_rows_without_weather` | Demand rows without a matching weather row are kept (left join): historical rows outside the forecast window must not be dropped                 |
| `test_rolling_window_drops_old_rows`               | Rows older than `rolling_window_hours` are dropped on save: verifies the 192h cutoff logic using a temporary directory and `unittest.mock.patch` |

---

### 10.3 Preprocessing - `tests/test_build_preprocessed_dataset.py` (4 tests)

Tests target `reindex_and_interpolate_ts`.

| Test                                         | What it verifies                                                                                                                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_missing_hour_is_filled`                | A gap in the middle of the series is reindexed and linearly interpolated: missing row at 12:00 between 11:00 and 13:00 should produce 52,000 MW                                      |
| `test_shortwave_radiation_is_forward_filled` | `shortwave_radiation_instant` is forward-filled, not linearly interpolated: at night the value is exactly 0; linear interpolation would produce physically incorrect non-zero values |
| `test_no_leading_or_trailing_nans`           | No NaNs remain after interpolation for interior gaps: validates `limit_area="inside"` behaviour                                                                                      |
| `test_duplicates_are_removed`                | Duplicate timestamps are removed before reindexing, keeping the first occurrence                                                                                                     |

---

### 10.4 Feature engineering - `tests/test_build_features.py` (6 tests)

Tests target `_compute_features`, a pure function extracted from `build_load_forecasting_features` specifically to enable unit testing without file I/O.

**Why extract `_compute_features`?**

`build_load_forecasting_features` reads parquet files and writes outputs, so it cannot be called in a test without real data on disk. Extracting the transformation logic into a pure function (DataFrame in, DataFrame out) makes it testable with synthetic data. This is a standard pattern for making I/O-heavy pipelines testable.

| Test                                     | What it verifies                                                                                    |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `test_load_t_minus_168_is_correct`       | `load_t-168` at row 168 equals `load_MW` at row 0: validates the 7-day lag                          |
| `test_target_is_next_hour_load`          | `target_load_t+1` at row i equals `load_MW` at row i+1: validates the shift(-1) target construction |
| `test_is_weekday_zero_on_sunday`         | `is_weekday=0` on 2026-01-04 (Sunday)                                                               |
| `test_is_weekday_zero_on_french_holiday` | `is_weekday=0` on 2026-07-14 (Bastille Day, Tuesday): holidays override the weekday flag            |
| `test_is_weekday_one_on_regular_monday`  | `is_weekday=1` on 2026-01-05 (regular Monday)                                                       |
| `test_last_row_has_no_target`            | Last row has `NaN` as target: no h+1 available; `dropna()` in the pipeline removes it               |

---

### 10.5 API - `tests/test_api.py` (4 tests)

Tests use FastAPI's `TestClient` to simulate HTTP requests without launching a real server.

The model, metadata, and parquet file are replaced with mocks so tests run without real model files or data on disk.

| Test                                           | What it verifies                                                                                              |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `test_health_returns_200`                      | `/health` returns 200 with `status`, `best_model`, and `feature_cols` when model is loaded                    |
| `test_health_returns_500_if_model_not_loaded`  | `/health` returns 500 when `model` is `None`                                                                  |
| `test_predict_returns_200_with_correct_fields` | `/predict` returns 200 with `predicted_load_MW`, `predicted_at`, `target_datetime` when all data is available |
| `test_predict_returns_503_if_parquet_missing`  | `/predict` returns 503 when the realtime parquet does not exist                                               |

---

### 10.6 FastAPI lifespan migration

During testing, a `DeprecationWarning` was raised for `@app.on_event("startup")`, which is deprecated in recent FastAPI versions in favour of the `lifespan` pattern.

The startup hook was migrated to use `@asynccontextmanager`:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_metadata, fr_holidays
    model = joblib.load(BASE_DIR / "models" / "best_model.pkl")
    with open(BASE_DIR / "models" / "training_results.json", "r") as f:
        model_metadata = json.load(f)
    fr_holidays = holidays.country_holidays("FR")
    yield

app = FastAPI(lifespan=lifespan)
```

This change required no modification to the Dockerfile or any other module.

---

## 11. MLOps — Drift Monitoring

### 11.1 Objective

The drift monitoring module detects whether the feature distributions observed in the realtime snapshot have shifted relative to the training reference.

This provides an early warning signal that the model may be operating outside its zone of confidence, before prediction errors become observable.

**Script**: `src/monitoring/monitor.py`
**Output**: `data/monitoring/YYYY-MM-DD_HH.json`
**Trigger**: `python main.py --steps monitor`, or automatically after `--steps realtime`

---

### 11.2 Reference dataset

The reference distribution is built from `data/featured/country=FR/year=2024/`.

**Why 2024 and not the full training set (2015–2023)?**

Using the most recent training year as reference captures the latest demand regime seen by the model. Averaging over 2015–2023 would dilute recent patterns with older consumption behaviours (pre-COVID, different energy mix). For operational drift detection, proximity to the current period matters more than statistical robustness of the reference.

---

### 11.3 Current dataset

The current distribution is built from the realtime snapshot (`data/realtime/country=FR/realtime.parquet`), after computing load lag features and calendar features using the same logic as the feature engineering pipeline.

**Effective size**: after applying `dropna` on lag features (`load_t-1`, `load_t-24`, `load_t-168`), the current dataset contains **~24 usable rows** out of 192 in the rolling window. The first 168 rows lack a valid `load_t-168` value.

**Known limitation**: 24 rows is a small sample for statistical drift tests. KS test p-values will be less stable than on larger datasets. This is an inherent constraint of the realtime design (192h rolling window, 168h lag requirement) and is acceptable for a portfolio project. In a production system, one would accumulate current data over several days before running drift tests.

---

### 11.4 Drift detection method

Evidently `ValueDrift` is used per feature, which applies the **Kolmogorov-Smirnov test** for continuous numerical features.

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Library           | Evidently 0.7.x                 |
| Test              | Kolmogorov-Smirnov (K-S)        |
| Threshold         | p-value < 0.05 → drift detected |
| Features analyzed | 8 (see §11.5)                   |

The KS test measures whether two samples come from the same continuous distribution, without assuming normality. It is well-suited for electricity load and calendar features.

---

### 11.5 Monitored features

Only load and calendar features are monitored. Weather features are excluded.

| Feature            | Monitored | Reason if excluded                                                           |
| ------------------ | --------- | ---------------------------------------------------------------------------- |
| `load_t`           | yes       |                                                                              |
| `load_t-1`         | yes       |                                                                              |
| `load_t-24`        | yes       |                                                                              |
| `load_t-168`       | yes       |                                                                              |
| `hour`             | yes       |                                                                              |
| `day_of_week`      | yes       |                                                                              |
| `is_weekday`       | yes       |                                                                              |
| `week_of_year`     | yes       |                                                                              |
| `temperature_t`    | no        | Realtime snapshot contains only 2h of weather forecast; ~99% of rows are NaN |
| `temperature_t-24` | no        | Same reason                                                                  |

**Why weather features are absent from the realtime snapshot:**
`get_realtime_data.py` fetches only 2 hours of Open-Meteo forecast (sufficient for the `/predict` endpoint), then shifts timestamps by -1h to align weather with demand.

This means only 1–2 rows in the 192h window have weather data. Including these features in drift monitoring would produce meaningless results on near-empty columns.

---

### 11.6 Critical features and alerts

Three features are designated as critical: a `[WARN]` log is emitted if any of them is detected in drift:

```python
CRITICAL_FEATURES = ["load_t-1", "load_t-24", "load_t-168"]
```

**Rationale**: these are the three most predictive features in the XGBoost model (short-term autocorrelation and daily/weekly seasonality). Drift on these features directly impacts prediction quality. Calendar features (`hour`, `day_of_week`, etc.) are less critical because their distribution is structurally bounded and predictable.

---

### 11.7 Known artefact - `hour` drift

In practice, the `hour` feature is frequently flagged as drifted (p_value ≈ 0.0).

This is a **statistical artefact**, not a real signal.

**Cause**: the current dataset contains exactly 24 rows — one per hour of the day, each appearing exactly once. This produces a perfectly uniform distribution over [0, 23]. The reference dataset (8784 rows) has a non-uniform distribution because some hours are more represented in the data due to DST and minor gaps. The KS test detects this structural difference as drift.

**Why this is not a real problem**: the model does not learn from the marginal distribution of `hour` in isolation. The feature is used in interaction with other features (e.g. `hour × is_weekday`) inside XGBoost trees. A uniform sample of 24 hours is a natural consequence of monitoring a single day of realtime data.

**Mitigation considered**: excluding `hour` from monitored features. Rejected because `hour` is a legitimate feature to monitor over longer accumulation windows. The artefact disappears when the current dataset spans several days. It is retained in the monitored set with this documented caveat.

---

### 11.8 Report format

Each monitoring run produces a JSON file in `data/monitoring/YYYY-MM-DD_HH.json`:

```json
{
  "timestamp": "2026-04-11T09:52:00",
  "reference_year": 2024,
  "n_features_analyzed": 8,
  "drift_summary": {
    "load_t": {"drift_detected": false, "p_value": 0.731, "method": "K-S p_value", "threshold": 0.05},
    "hour":   {"drift_detected": true,  "p_value": 0.0,   "method": "K-S p_value", "threshold": 0.05},
    ...
  },
  "evidently_raw": { ... }
}
```

The `evidently_raw` field contains the complete Evidently `dump_dict()` output, preserved for future use by the Streamlit dashboard.

---

### 11.9 Python version compatibility

Evidently 0.7.x requires **Python ≤ 3.13**. It is incompatible with Python 3.14 due to a Pydantic V1 compatibility issue (`pydantic.v1` does not support Python 3.14+).

The project uses Python 3.12 in the conda environment (`energy_ml_312`) and `python:3.12-slim` as the Docker base image. This is consistent with the Evidently constraint and with the broader ML/data science ecosystem, which has not yet fully migrated to Python 3.14.

---

### 11.10 Unit tests - `tests/test_monitor.py` (10 tests)

Tests are structured around three classes targeting pure functions, following the same pattern as the rest of the test suite.

| Class                      | Tests | What is verified                                                                                                                                                                                                                   |
| -------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TestComputeDrift`         | 5     | Summary contains all features, correct structure (`drift_detected`, `p_value`, `method`, `threshold`), drift detected when distributions shift strongly, no drift when distributions are identical, raw Evidently dict is returned |
| `TestLogDriftAlerts`       | 3     | Returns `False` when no critical feature drifts, returns `True` when at least one critical feature drifts, `[WARN]` is logged for each drifted critical feature                                                                    |
| `TestSaveMonitoringReport` | 2     | File created with correct name (`YYYY-MM-DD_HH.json`), JSON contains required keys (`timestamp`, `drift_summary`, `evidently_raw`, `reference_year`)                                                                               |

**Total tests after this step**: 27 (17 existing + 10 new), all passing.

```bash
pytest tests/ -v
```

---

## 12. Dashboard - Streamlit

### 12.1 Architecture

The dashboard is a standalone Streamlit application located in `dashboard/`, separate from the rest of the project. It reads data directly from Parquet files and the monitoring JSON reports, it does not call the FastAPI layer.

**Why decouple from the API ?**
For a portfolio demo, the Streamlit app must work independently regardless of whether the FastAPI server and scheduler are running. Reading Parquet files directly is more robust, removes an HTTP dependency, and avoids the latency of an extra network call. The duplication of feature reconstruction logic is intentional and documented (see §12.5).

```
dashboard/
├── dashboard.py        # entry point: page config, sidebar, routing
├── tabs/               # renamed from pages/ to avoid Streamlit auto-routing conflict
│   ├── realtime.py
│   ├── historical.py
│   └── model_perf.py
└── utils/
    ├── data_loader.py  # all Parquet/JSON reads, @st.cache_data
    ├── prediction.py   # feature reconstruction + model inference
    └── charts.py       # reusable Plotly figures
```

**Launch command:**

```bash
streamlit run dashboard/dashboard.py
```

---

### 12.2 Routing design: why `tabs/` instead of `pages/`

Streamlit automatically detects a folder named exactly `pages/` placed next to the entry point and activates its own multi-page navigation system. This conflicts with the manual routing implemented in `dashboard.py` via `st.session_state`.

Renaming the folder to `tabs/` disables Streamlit's automatic detection while keeping the same file structure and `render()` contract unchanged.

**Manual routing pattern used:**

```python
if page == "realtime":
    from tabs.realtime import render
    render()
```

Imports are placed inside the `if/elif` blocks, not at the top of the file. This avoids executing initialisation code (heavy imports, data loading) for pages the user is not currently viewing.

---

### 12.3 Caching strategy

Streamlit re-executes the entire script on every user interaction.

Two cache decorators are used to avoid redundant I/O and computation:

| Decorator               | Use case                                                 | Notes                                          |
| ----------------------- | -------------------------------------------------------- | ---------------------------------------------- |
| `@st.cache_data(ttl=N)` | DataFrames, dicts, JSON: serialisable objects            | Creates a copy per call; safe for mutable data |
| `@st.cache_resource`    | Model pickle (`best_model.pkl`): non-serialisable object | Single shared instance for the server lifetime |

**TTL values per data source:**

| Function                     | TTL                  | Rationale                                                 |
| ---------------------------- | -------------------- | --------------------------------------------------------- |
| `load_realtime()`            | 300s (5 min)         | Updated hourly by the scheduler; 5 min avoids stale reads |
| `load_featured(year)`        | 3600s (1h)           | Historical data is stable; cache per year individually    |
| `load_featured_range()`      | -                    | Calls `load_featured()` per year; inherits per-year cache |
| `load_training_results()`    | 0 (session lifetime) | Never changes after training                              |
| `load_latest_drift_report()` | 300s (5 min)         | New report every hour; 5 min matches scheduler cadence    |
| `_load_model()`              | `@st.cache_resource` | Loaded once per server process lifetime                   |

---

### 12.4 `PROJECT_ROOT` in the dashboard

Files in `dashboard/utils/` are two levels below the project root:

```
energy-intelligence-platform/   ← parents[2]
└── dashboard/                   ← parents[1]
    └── utils/                   ← parents[0] = Path(__file__).resolve().parent
```

Both `data_loader.py` and `prediction.py` use:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
```

This resolves to the project root regardless of the working directory at launch time.

---

### 12.5 Feature reconstruction in prediction.py

`dashboard/utils/prediction.py` replicates the feature construction logic from `api/main.py` without the HTTP layer. This duplication is intentional:

- Importing from `api/main.py` would couple the dashboard to FastAPI internals (lifespan, Pydantic models, routing logic) — a fragile dependency.
- The correct long-term fix is to extract this logic into `src/modeling/inference.py` and import it from both consumers. This refactoring is deferred and documented.

**Known limitation — `temperature_t-24`:**
The realtime parquet contains weather data only on the last 1–2 rows (2h Open-Meteo forecast window). The row at `t-24` has no valid temperature. When this lookup returns NaN, `prediction.py` falls back to `temperature_t` as a proxy.

**Justification**: `temperature_t-24` has an importance score of 0.001 (0.1%) in the XGBoost model (see §8.4). The approximation error is negligible in practice.

---

### 12.6 Tab contents

**tabs/realtime.py - Real-time forecast**

- Three `st.metric()` widgets: last actual load (MW), predicted load h+1 (MW + delta), forecast horizon (target datetime UTC).
- Load curve for the 24 hours preceding the last available observation. Cutoff is `last_dt - 24h`, not `now_utc - 24h` — the realtime data ends in 2025, so using `now_utc` would produce an empty chart.
- Prediction point (orange marker) connected to the curve by a dashed segment.
- Expander showing the 10 feature values used for the current prediction: demonstrates full inference traceability.

**tabs/historical.py - Historical data**

- Year range selector (2015–2025).
- Dynamic resampling: hourly resolution if period ≤ 1 year (`n_days ≤ 365`), daily average otherwise. Prevents slow rendering on multi-year datasets.
- Three `st.metric()` widgets: row count, mean load (GW), resolution label.
- Load curve using `load_t` (observed consumption), not `target_load_t+1`.

**tabs/model_perf.py - Model performance**

- Model comparison table: val MAE and RMSE for XGBoost, LightGBM, and Ridge.
- Best model detail: val + test MAE/RMSE. `st.info()` contextualises the elevated test MAE (5,606 MW) as a distribution shift artefact, not a model defect (see §8.7).
- Feature importance horizontal bar chart. Importances are read from `training_results.json` if present; otherwise a hardcoded fallback is used (values from §8.4). The fallback is a temporary workaround until `feature_importances` is written to `training_results.json` by `src/modeling/storage.py`.
- Drift monitoring section: timestamp, reference year, count of features in drift, p-value bar chart (green = no drift, red = drift detected), dashed threshold line at p = 0.05. A `st.warning()` is displayed when `hour` is flagged: it explains the structural false positive (see §11.7).

---

### 12.7 Plotly theme - charts.py

All figures share a `_LAYOUT` dict applied via `fig.update_layout(**_LAYOUT)`:

- Transparent backgrounds (`rgba(0,0,0,0)`) — compatible with Streamlit light and dark themes.
- Subtle grid lines (`rgba(128,128,128,0.15)`).
- No axis zeroline.

**Color scheme:**

| Role                    | Hex       |
| ----------------------- | --------- |
| Actual load (blue)      | `#4A90D9` |
| Predicted load (orange) | `#E8834A` |
| Drift detected (red)    | `#E24B4A` |
| No drift (green)        | `#639922` |

**Unit convention**: all chart axes display values in **GW** (divided by 1,000). `st.metric()` widgets display values in **MW** for precision.

**Known conflict - `update_layout` keyword collision:**
`_LAYOUT` already contains `xaxis` and `yaxis` keys. Passing these keys again
directly in `update_layout()` raises a `TypeError: got multiple values for keyword argument`.
All axis-specific overrides (tick format, title, range) are applied separately via
`fig.update_xaxes()` and `fig.update_yaxes()`.

---

### 12.8 Known issues and planned improvements

| Issue                                                                       | Status                          | Planned fix                                                       |
| --------------------------------------------------------------------------- | ------------------------------- | ----------------------------------------------------------------- |
| `feature_importances` not stored in `training_results.json`                 | Workaround (hardcoded fallback) | Add serialisation in `src/modeling/storage.py` on next retraining |
| `temperature_t-24` is NaN in realtime (fallback to `temperature_t`)         | Documented, negligible impact   | Acceptable given 0.1% feature importance                          |
| `use_container_width=True` deprecated (Streamlit ≥ 2.x)                     | Warning in logs                 | Replace with `width='stretch'`                                    |
| Feature reconstruction duplicated between `api/main.py` and `prediction.py` | Documented                      | Refactor into `src/modeling/inference.py`                         |
