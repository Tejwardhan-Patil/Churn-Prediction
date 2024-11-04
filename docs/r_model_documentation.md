# R Model Documentation

## Overview

The Churn Prediction system integrates R-based statistical models for churn prediction. The R models complement the machine learning models developed in Python.

### Models

- **GLM**: Logistic Regression model using the `glmnet` package.
- **Random Forest**: Built using the `randomForest` package in R.

## Files

- `r_models.R`: Contains the model definitions.
- `evaluate.R`: Evaluation scripts including AUC-ROC and performance metrics.
- `test_models.R`: Unit tests for the R models.

## Deployment

- The R models are served using the R Plumber API (`app.R`) and containerized with Docker.
