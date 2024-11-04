# System Architecture

## Overview

The Churn Prediction system integrates Python and R scripts for data preprocessing, feature engineering, model development, and deployment. The system is designed to be modular and scalable, with support for both machine learning and statistical models. It is composed of multiple layers, including data ingestion, preprocessing, model training, evaluation, deployment, and monitoring.

### Components

- **Data Layer**:
  - Raw data is ingested in `.csv` format and stored in the `/data/raw/` directory.
  - Preprocessing scripts in Python (`preprocess.py`) and R (`preprocess.R`) handle data cleaning and preparation.
  - Feature engineering occurs in both Python (`feature_engineering.py`) and R (`feature_engineering.R`).

- **Model Layer**:
  - Machine learning models (e.g., Logistic Regression, Random Forest, XGBoost, Neural Networks) are developed using Python (`baseline_model.py`, `advanced_models.py`).
  - Statistical models are implemented in R (`r_models.R`), which include GLM and Random Forest.
  - Models are evaluated and tuned using Python (`hyperparameter_tuning.py`) and R (`evaluate.R`).

- **Deployment Layer**:
  - The system offers two APIs: one built using Python's Flask/FastAPI (`app.py`) and one in R Plumber (`app.R`).
  - Docker is used to containerize the APIs and models, with both Python and R services.

- **Monitoring Layer**:
  - The monitoring system includes tracking model drift (`model_drift.py`) and generating performance reports using both Python and R.
  - Grafana dashboards (`grafana_dashboards.json`) provide real-time monitoring.
