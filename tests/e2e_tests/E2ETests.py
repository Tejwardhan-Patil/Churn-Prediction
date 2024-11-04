import pytest
import requests
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants for the test environment
API_URL = "http://localhost:5000/api/predict"
MODEL_PATH = "models/saved_models/best_model.pkl"
CLEANED_DATA_PATH = "data/processed/cleaned_customer_data.csv"

@pytest.fixture(scope="module")
def load_cleaned_data():
    """Load the cleaned customer data for testing."""
    if not os.path.exists(CLEANED_DATA_PATH):
        pytest.fail(f"Cleaned data file not found at {CLEANED_DATA_PATH}")
    data = pd.read_csv(CLEANED_DATA_PATH)
    return data

@pytest.fixture(scope="module")
def load_model():
    """Load the saved model for testing."""
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"Model not found at {MODEL_PATH}")
    model = pd.read_pickle(MODEL_PATH)
    return model

def test_api_response():
    """Test if the API is up and running."""
    response = requests.get(API_URL)
    assert response.status_code == 200, "API is not reachable"

def test_predict_api(load_cleaned_data):
    """Test predictions from the deployed API."""
    test_sample = load_cleaned_data.sample(5).to_dict(orient="records")
    response = requests.post(API_URL, json={"data": test_sample})
    assert response.status_code == 200, "Prediction API failed"
    predictions = response.json().get("predictions", [])
    assert len(predictions) == 5, "API did not return correct number of predictions"

def test_model_accuracy(load_model, load_cleaned_data):
    """Test model accuracy on the cleaned data."""
    X = load_cleaned_data.drop(columns=["churn"])
    y_true = load_cleaned_data["churn"]
    y_pred = load_model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    assert accuracy > 0.7, f"Accuracy is too low: {accuracy}"
    assert precision > 0.6, f"Precision is too low: {precision}"
    assert recall > 0.6, f"Recall is too low: {recall}"
    assert f1 > 0.65, f"F1 Score is too low: {f1}"

def test_model_drift():
    """Test if model drift detection script runs successfully."""
    os.system("python monitoring/metrics/model_drift.py")
    assert os.path.exists("monitoring/metrics/drift_report.json"), "Model drift report not generated"

def test_feature_engineering_pipeline():
    """Test the feature engineering pipeline."""
    os.system("python data/scripts/feature_engineering.py")
    assert os.path.exists("data/processed/cleaned_customer_data.csv"), "Feature engineering output missing"

def test_hyperparameter_tuning():
    """Test the hyperparameter tuning script."""
    os.system("python models/hyperparameter_tuning.py")
    assert os.path.exists("models/tuning_results.json"), "Hyperparameter tuning did not produce results"

def test_docker_container():
    """Test if the Docker container is up and running."""
    container_status = os.popen("docker ps | grep churn_prediction_api").read()
    assert "churn_prediction_api" in container_status, "Docker container for API not running"

def test_model_evaluation_metrics():
    """Test model evaluation metrics after training."""
    os.system("python models/model_evaluation.py")
    assert os.path.exists("models/evaluation_report.json"), "Model evaluation report missing"

def test_end_to_end_pipeline():
    """Test the complete end-to-end pipeline."""
    os.system("python pipeline/airflow/dags/churn_prediction_dag.py")
    assert os.path.exists("data/processed/cleaned_customer_data.csv"), "Data processing failed"
    assert os.path.exists(MODEL_PATH), "Model not trained"
    assert os.path.exists("models/evaluation_report.json"), "Model evaluation failed"

if __name__ == "__main__":
    pytest.main()