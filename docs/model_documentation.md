# Model Documentation

## Overview

This document describes the various machine learning models used in the Churn Prediction project, including their development, evaluation, and selection processes.

### Model Pipeline

1. **Baseline Model**: Logistic Regression implemented in `baseline_model.py`.
2. **Advanced Models**:
   - Random Forest (`advanced_models.py`)
   - XGBoost (`advanced_models.py`)
   - Neural Networks (`advanced_models.py`)

### Hyperparameter Tuning

- The tuning process is handled by `hyperparameter_tuning.py` using GridSearch or Optuna for model optimization.

### Model Evaluation

- The models are evaluated using metrics such as Precision, Recall, F1-Score, and AUC-ROC. Evaluation scripts can be found in `model_evaluation.py`.

## Files

- `baseline_model.py`
- `advanced_models.py`
- `hyperparameter_tuning.py`
- `evaluate.R` (for R models)
