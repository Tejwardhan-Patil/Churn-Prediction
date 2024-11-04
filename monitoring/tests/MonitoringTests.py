import unittest
import os
import json
from monitoring.metrics import model_drift, monitor
from monitoring.logging import log_config
from monitoring.alerts import alert_rules

class TestModelDrift(unittest.TestCase):

    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(__file__), '../data/processed/cleaned_customer_data.csv')
        self.model_path = os.path.join(os.path.dirname(__file__), '../models/saved_models/best_model.pkl')
        self.threshold = 0.05  # Drift threshold for the test

    def test_model_drift_detection(self):
        drift_score = model_drift.detect_drift(self.test_data_path, self.model_path, threshold=self.threshold)
        self.assertLess(drift_score, self.threshold, "Model drift exceeds acceptable threshold")

    def test_model_drift_with_insufficient_data(self):
        # Simulate a scenario with insufficient data
        empty_data_path = os.path.join(os.path.dirname(__file__), '../data/processed/empty_customer_data.csv')
        with self.assertRaises(ValueError):
            model_drift.detect_drift(empty_data_path, self.model_path, threshold=self.threshold)

    def test_model_drift_invalid_model(self):
        # Test with an invalid model path
        invalid_model_path = os.path.join(os.path.dirname(__file__), '../models/saved_models/invalid_model.pkl')
        with self.assertRaises(FileNotFoundError):
            model_drift.detect_drift(self.test_data_path, invalid_model_path, threshold=self.threshold)


class TestPerformanceMetrics(unittest.TestCase):

    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(__file__), '../data/processed/cleaned_customer_data.csv')
        self.model_path = os.path.join(os.path.dirname(__file__), '../models/saved_models/best_model.pkl')

    def test_monitor_accuracy(self):
        metrics = monitor.evaluate_performance(self.test_data_path, self.model_path)
        self.assertGreaterEqual(metrics['accuracy'], 0.8, "Accuracy is lower than expected")

    def test_monitor_precision(self):
        metrics = monitor.evaluate_performance(self.test_data_path, self.model_path)
        self.assertGreaterEqual(metrics['precision'], 0.75, "Precision is lower than expected")

    def test_monitor_recall(self):
        metrics = monitor.evaluate_performance(self.test_data_path, self.model_path)
        self.assertGreaterEqual(metrics['recall'], 0.7, "Recall is lower than expected")

    def test_monitor_f1_score(self):
        metrics = monitor.evaluate_performance(self.test_data_path, self.model_path)
        self.assertGreaterEqual(metrics['f1_score'], 0.75, "F1 score is lower than expected")

    def test_monitor_performance_with_invalid_data(self):
        # Test with invalid data path
        invalid_data_path = os.path.join(os.path.dirname(__file__), '../data/processed/invalid_customer_data.csv')
        with self.assertRaises(FileNotFoundError):
            monitor.evaluate_performance(invalid_data_path, self.model_path)


class TestAlertRules(unittest.TestCase):

    def setUp(self):
        self.alert_config_path = os.path.join(os.path.dirname(__file__), '../monitoring/alerts/alert_rules.yaml')

    def test_load_alert_rules(self):
        # Test loading of alert rules
        with open(self.alert_config_path, 'r') as f:
            alert_config = json.load(f)
        self.assertIsNotNone(alert_config, "Failed to load alert rules")

    def test_alert_rule_threshold(self):
        # Ensure alert rules contain a threshold for alerts
        with open(self.alert_config_path, 'r') as f:
            alert_config = json.load(f)
        self.assertIn('threshold', alert_config, "Alert rules missing threshold")

    def test_invalid_alert_rule(self):
        # Test invalid alert rule configuration
        invalid_alert_config_path = os.path.join(os.path.dirname(__file__), '../monitoring/alerts/invalid_alert_rules.yaml')
        with self.assertRaises(json.JSONDecodeError):
            with open(invalid_alert_config_path, 'r') as f:
                json.load(f)


class TestLogging(unittest.TestCase):

    def setUp(self):
        self.log_path = os.path.join(os.path.dirname(__file__), '../monitoring/logging/log_config.py')

    def test_logging_setup(self):
        logger = log_config.setup_logging('test_logger')
        self.assertIsNotNone(logger, "Logging setup failed")

    def test_log_file_exists(self):
        log_file = os.path.join(os.path.dirname(__file__), '../logs/monitoring.log')
        self.assertTrue(os.path.exists(log_file), "Log file does not exist")

    def test_log_error_handling(self):
        logger = log_config.setup_logging('test_logger')
        with self.assertLogs(logger, level='ERROR') as cm:
            logger.error('Test error message')
        self.assertIn('ERROR', cm.output[0], "Error logging failed")

    def test_log_info_handling(self):
        logger = log_config.setup_logging('test_logger')
        with self.assertLogs(logger, level='INFO') as cm:
            logger.info('Test info message')
        self.assertIn('INFO', cm.output[0], "Info logging failed")


class TestMonitoringIntegration(unittest.TestCase):

    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(__file__), '../data/processed/cleaned_customer_data.csv')
        self.model_path = os.path.join(os.path.dirname(__file__), '../models/saved_models/best_model.pkl')
        self.alert_config_path = os.path.join(os.path.dirname(__file__), '../monitoring/alerts/alert_rules.yaml')

    def test_monitoring_end_to_end(self):
        # Load alert rules
        with open(self.alert_config_path, 'r') as f:
            alert_config = json.load(f)
        
        # Evaluate model performance
        metrics = monitor.evaluate_performance(self.test_data_path, self.model_path)

        # Ensure no alert is triggered if metrics are within bounds
        if metrics['f1_score'] < alert_config['threshold']['f1_score']:
            alert_triggered = True
        else:
            alert_triggered = False

        self.assertFalse(alert_triggered, "Alert triggered unexpectedly")

    def test_monitoring_with_drift(self):
        # Simulate model drift scenario
        drift_score = model_drift.detect_drift(self.test_data_path, self.model_path, threshold=0.05)
        alert_config = {'threshold': {'drift_score': 0.04}}

        if drift_score > alert_config['threshold']['drift_score']:
            alert_triggered = True
        else:
            alert_triggered = False

        self.assertTrue(alert_triggered, "Alert for model drift was not triggered")


if __name__ == '__main__':
    unittest.main()