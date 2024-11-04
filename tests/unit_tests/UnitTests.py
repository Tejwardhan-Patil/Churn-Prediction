import unittest
import pandas as pd
import subprocess
import json
from data.scripts.preprocess import preprocess_data
from data.scripts.feature_engineering import create_features, select_features
from models.baseline_model import LogisticRegressionModel
from models.advanced_models import RandomForestModel, XGBoostModel
from models.model_evaluation import evaluate_model
from deployment.api.app import app as flask_app

class PreprocessingTests(unittest.TestCase):

    def setUp(self):
        # Load data from CSV file
        self.raw_data_path = 'data/processed/cleaned_customer_data.csv'
        self.raw_data = pd.read_csv(self.raw_data_path)

    def test_preprocess_data(self):
        cleaned_data = preprocess_data(self.raw_data)
        # Test if cleaning process works on the loaded CSV data
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertIn('customer_id', cleaned_data.columns)
        self.assertIn('age', cleaned_data.columns)

class FeatureEngineeringTests(unittest.TestCase):

    def setUp(self):
        # Load cleaned data from CSV file
        self.cleaned_data_path = 'data/processed/cleaned_customer_data.csv'
        self.cleaned_data = pd.read_csv(self.cleaned_data_path)

    def test_create_features(self):
        features = create_features(self.cleaned_data)
        self.assertIn('age_group', features.columns)
        self.assertGreater(len(features['customer_id']), 0)

    def test_select_features(self):
        features = create_features(self.cleaned_data)
        selected_features = select_features(features)
        self.assertGreater(len(selected_features.columns), 0)

class ModelTests(unittest.TestCase):

    def setUp(self):
        # Load feature data from CSV file
        self.features_path = 'data/processed/features.csv'
        self.feature_data = pd.read_csv(self.features_path)
        self.labels = self.feature_data['churn'].values

    def test_logistic_regression(self):
        model = LogisticRegressionModel()
        model.fit(self.feature_data.drop(columns=['churn']), self.labels)
        predictions = model.predict([[25], [40]])
        self.assertEqual(len(predictions), 2)
        self.assertIn(0, predictions)

    def test_random_forest(self):
        model = RandomForestModel()
        model.fit(self.feature_data.drop(columns=['churn']), self.labels)
        predictions = model.predict([[25], [40]])
        self.assertEqual(len(predictions), 2)
        self.assertIn(0, predictions)

    def test_xgboost(self):
        model = XGBoostModel()
        model.fit(self.feature_data.drop(columns=['churn']), self.labels)
        predictions = model.predict([[25], [40]])
        self.assertEqual(len(predictions), 2)
        self.assertIn(0, predictions)

class EvaluationTests(unittest.TestCase):

    def setUp(self):
        self.predictions = [0, 1, 0]
        self.true_labels = [0, 1, 0]

    def test_evaluate_model(self):
        report = evaluate_model(self.true_labels, self.predictions)
        self.assertIn('accuracy', report)
        self.assertIn('precision', report)
        self.assertGreaterEqual(report['accuracy'], 0.0)

class APITests(unittest.TestCase):

    def setUp(self):
        flask_app.config['TESTING'] = True
        self.app = flask_app.test_client()

    def test_predict_route(self):
        response = self.app.post(
            '/predict',
            data=json.dumps({'age': 35, 'income': 50000}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('churn_prediction', data)

    def test_health_check_route(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')

class IntegrationTests(unittest.TestCase):

    def test_full_pipeline(self):
        # Load raw data from CSV
        raw_data = pd.read_csv('data/processed/cleaned_customer_data.csv')

        # Preprocess data
        cleaned_data = preprocess_data(raw_data)

        # Feature Engineering
        features = create_features(cleaned_data)

        # Train a Logistic Regression Model
        model = LogisticRegressionModel()
        model.fit(features.drop(columns=['churn']), features['churn'])

        # Make predictions
        predictions = model.predict([[30]])

        # Evaluate
        evaluation = evaluate_model([0], predictions)
        self.assertGreaterEqual(evaluation['accuracy'], 0.0)

class MonitoringTests(unittest.TestCase):

    def test_drift_detection(self):
        from monitoring.metrics.model_drift import detect_drift
        current_data = [0.2, 0.4, 0.3, 0.5]
        past_data = [0.1, 0.2, 0.1, 0.3]
        drift_detected = detect_drift(current_data, past_data)
        self.assertFalse(drift_detected)

    def test_monitor_performance(self):
        # Call R script for monitoring performance
        result = subprocess.run(
            ['Rscript', 'monitoring/metrics/monitor.R'],
            capture_output=True,
            text=True
        )
        self.assertIn("Performance Metrics", result.stdout)

class SecurityTests(unittest.TestCase):

    def test_jwt_auth(self):
        from security.authentication.jwt_auth import generate_token, verify_token
        token = generate_token('user1')
        self.assertIsNotNone(token)
        verified = verify_token(token)
        self.assertTrue(verified)

    def test_data_encryption(self):
        from security.encryption.data_encryption import encrypt_data, decrypt_data
        original_data = 'Sensitive data'
        encrypted = encrypt_data(original_data)
        decrypted = decrypt_data(encrypted)
        self.assertEqual(original_data, decrypted)

if __name__ == '__main__':
    unittest.main()