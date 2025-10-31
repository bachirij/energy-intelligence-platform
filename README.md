# ⚡ Energy Demand Forecasting – End-to-End AI/ML Engineering Project

## Overview
This repository presents a complete **AI/ML engineering project** focused on **hourly electricity demand forecasting (h+24)** in France.  
The goal is to build a **scalable, reproducible, and MLOps-ready system** from data collection to model deployment and monitoring.

---

## Project Scope
- **Forecasting:** Day-ahead electricity demand at hourly resolution  
- **Data sources:** ENTSO-E transparency platform (demand), Open Meteo (weather), holidays, temporal features  
- **Models:** Classical ML (Random Forest, XGBoost), Time Series (SARIMA, Prophet), Deep Learning (LSTM, GRU, Transformer)  
- **MLOps:** Data pipelines (Airflow/Prefect + DVC), experiment tracking (MLflow), model serving (FastAPI), deployment (Docker + CI/CD), monitoring (Evidently + Streamlit)  
- **Dashboard:** Interactive visualization of historical data and forecasts

---

## Objectives
- Automate data ingestion and preprocessing  
- Compare and evaluate multiple forecasting models  
- Serve predictions via a REST API  
- Deploy a cloud-hosted, scalable solution  
- Provide a dashboard for monitoring predictions and KPIs

---

## Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, Scikit-Learn, PyTorch, XGBoost, Statsmodels, MLflow, DVC  
- **Tools & Frameworks:** Airflow/Prefect, FastAPI, Docker, CI/CD, Streamlit  
- **Cloud:** AWS / GCP / HuggingFace Spaces

---

## Users & Impact
Intended for **grid operators** and energy analysts to:  
- Anticipate consumption peaks  
- Optimize production & grid management  
- Reduce operational costs and support renewable integration  
- Enable smart grid applications and predictive planning

---

## License
MIT License 
