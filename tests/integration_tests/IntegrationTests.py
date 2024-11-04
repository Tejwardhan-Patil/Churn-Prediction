import unittest
import requests
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from models.baseline_model import LogisticRegressionModel
from models.advanced_models import RandomForestModel, XGBoostModel
from deployment.api.app import app as flask_app
from data.scripts.preprocess import preprocess_data
from features.feature_creation import generate_features
from models.model_selection import ModelSelector

class IntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup paths and environment variables
        cls.raw_data_path = os.path.join(os.getcwd(), 'data/raw/customer_data.csv')
        cls.processed_data_path = os.path.join(os.getcwd(), 'data/processed/cleaned_customer_data.csv')
        cls.model_save_path = os.path.join(os.getcwd(), 'models/saved_models/best_model.pkl')
        cls.api_url = "http://localhost:5000/predict"
        cls.flask_app = flask_app.test_client()

    def test_data_preprocessing(self):
        """ Test integration of data preprocessing """
        raw_data = pd.read_csv(self.raw_data_path)
        cleaned_data = preprocess_data(raw_data)
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertGreater(len(cleaned_data), 0)

    def test_feature_creation(self):
        """ Test feature engineering integration """
        cleaned_data = pd.read_csv(self.processed_data_path)
        features, labels = generate_features(cleaned_data)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)

    def test_model_training(self):
        """ Test the integration of model training """
        cleaned_data = pd.read_csv(self.processed_data_path)
        features, labels = generate_features(cleaned_data)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
        
        # Train logistic regression as baseline
        logistic_model = LogisticRegressionModel()
        logistic_model.train(X_train, y_train)
        baseline_accuracy = logistic_model.evaluate(X_test, y_test)
        
        # Train random forest
        rf_model = RandomForestModel()
        rf_model.train(X_train, y_train)
        rf_accuracy = rf_model.evaluate(X_test, y_test)

        # Train XGBoost
        xgb_model = XGBoostModel()
        xgb_model.train(X_train, y_train)
        xgb_accuracy = xgb_model.evaluate(X_test, y_test)
        
        self.assertGreater(baseline_accuracy, 0.5)
        self.assertGreater(rf_accuracy, baseline_accuracy)
        self.assertGreater(xgb_accuracy, rf_accuracy)

    def test_model_selection(self):
        """ Test integration of model selection logic """
        cleaned_data = pd.read_csv(self.processed_data_path)
        features, labels = generate_features(cleaned_data)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
        
        model_selector = ModelSelector()
        best_model, best_accuracy = model_selector.select_best_model(X_train, y_train, X_test, y_test)
        
        self.assertIsNotNone(best_model)
        self.assertGreater(best_accuracy, 0.5)

    def test_api_deployment(self):
        """ Test model API deployment """
        # Load a payload
        payload = {
            "customer_id": "12345",
            "features": {
                "age": 35,
                "balance": 1000,
                "num_of_products": 2,
                "has_credit_card": 1,
                "is_active_member": 0,
                "estimated_salary": 80000
            }
        }
        
        response = self.flask_app.post(self.api_url, json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json)
        self.assertIn("probability", response.json)
        
    def test_end_to_end(self):
        """ End-to-end test: from preprocessing to model API """
        raw_data = pd.read_csv(self.raw_data_path)
        
        # Preprocess data
        cleaned_data = preprocess_data(raw_data)
        features, labels = generate_features(cleaned_data)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

        # Train and select the best model
        model_selector = ModelSelector()
        best_model, _ = model_selector.select_best_model(X_train, y_train, X_test, y_test)
        
        # Serialize the model
        with open(self.model_save_path, 'wb') as model_file:
            best_model.save(model_file)
        
        # Test API prediction using serialized model
        payload = {
            "customer_id": "54321",
            "features": {
                "age": 45,
                "balance": 5000,
                "num_of_products": 1,
                "has_credit_card": 0,
                "is_active_member": 1,
                "estimated_salary": 120000
            }
        }
        
        response = self.flask_app.post(self.api_url, json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json)
        self.assertIn("probability", response.json)

if __name__ == '__main__':
    unittest.main()