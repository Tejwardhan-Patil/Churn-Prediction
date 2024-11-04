# API Documentation

## Overview

The Churn Prediction API provides endpoints for serving machine learning models and statistical models, handling both Python and R-based models. The API is built using Flask/FastAPI for Python and Plumber for R.

## Endpoints

### POST /predict

- **Description**: Predict churn for a customer using the deployed model.
- **Request**:
  - Content-Type: `application/json`
  - Body:

    ```json
    {
      "customer_data": {...}
    }
    ```

- **Response**:
  
  ```json
  {
    "prediction": "churn" or "no_churn",
    "probability": 0.87
  }
  ```

### GET /model_info

- **Description**: Get details about the currently deployed model.
- **Response**:
  
  ```json
  {
    "model": "Random Forest",
    "version": "v1.2",
    "metrics": {
      "accuracy": 0.91,
      "f1_score": 0.88
    }
  }
  ```

## Authentication

- JWT-based authentication is implemented using the `jwt_auth.py` script for securing the API.
