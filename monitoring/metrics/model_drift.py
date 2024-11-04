import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load configuration for model drift detection (thresholds, paths, etc)
CONFIG_PATH = "/mnt/data/configs/config.prod.yaml"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Email alert configuration
EMAIL_HOST = "smtp.website.com"
EMAIL_PORT = 587
EMAIL_USERNAME = "alert@website.com"
EMAIL_PASSWORD = "strongpassword"
ALERT_RECIPIENT = "admin@website.com"

# Drift detection class
class ModelDriftDetector:
    def __init__(self, model_path, baseline_metrics_path, threshold_config):
        self.model_path = model_path
        self.baseline_metrics_path = baseline_metrics_path
        self.threshold_config = threshold_config
        self.load_model()
        self.load_baseline_metrics()

    def load_model(self):
        logger.info("Loading the model from path: %s", self.model_path)
        try:
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
        except Exception as e:
            logger.error("Error loading the model: %s", str(e))
            raise

    def load_baseline_metrics(self):
        logger.info("Loading baseline metrics from path: %s", self.baseline_metrics_path)
        try:
            self.baseline_metrics = pd.read_csv(self.baseline_metrics_path)
        except Exception as e:
            logger.error("Error loading baseline metrics: %s", str(e))
            raise

    def calculate_metrics(self, X_test, y_test):
        logger.info("Calculating model metrics")
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        logger.info("Calculated metrics: %s", metrics)
        return metrics

    def compare_metrics(self, current_metrics):
        logger.info("Comparing current metrics with baseline metrics")
        drift_detected = False
        for metric, value in current_metrics.items():
            baseline_value = self.baseline_metrics[metric].values[0]
            threshold = self.threshold_config.get(metric, 0.05)
            if abs(value - baseline_value) > threshold:
                logger.warning(f"Significant drift detected in {metric}: baseline={baseline_value}, current={value}")
                drift_detected = True
        return drift_detected

    def save_metrics(self, metrics, output_path):
        logger.info("Saving the latest metrics")
        try:
            pd.DataFrame([metrics]).to_csv(output_path, index=False)
        except Exception as e:
            logger.error("Error saving metrics: %s", str(e))
            raise

    def send_alert(self, drift_metric, current_value, baseline_value):
        logger.info(f"ALERT: Model drift detected in {drift_metric}. Current: {current_value}, Baseline: {baseline_value}")
        
        # Compose the email
        subject = f"Model Drift Alert: {drift_metric}"
        body = f"""\
        <html>
            <body>
                <p>Dear Admin,</p>
                <p>A model drift has been detected in the {drift_metric} metric.</p>
                <p>Baseline Value: {baseline_value}<br>
                   Current Value: {current_value}<br>
                   Threshold Exceeded: Yes</p>
                <p>Please take the necessary actions.</p>
            </body>
        </html>
        """
        
        # Set up the email parameters
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USERNAME
        msg['To'] = ALERT_RECIPIENT
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        # Send the email
        try:
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
                server.starttls()
                server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
                server.sendmail(EMAIL_USERNAME, ALERT_RECIPIENT, msg.as_string())
                logger.info(f"Alert email sent successfully for {drift_metric}.")
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            raise

    def monitor(self, X_test, y_test, metrics_output_path):
        current_metrics = self.calculate_metrics(X_test, y_test)
        drift_detected = self.compare_metrics(current_metrics)
        if drift_detected:
            for metric, value in current_metrics.items():
                baseline_value = self.baseline_metrics[metric].values[0]
                if abs(value - baseline_value) > self.threshold_config.get(metric, 0.05):
                    self.send_alert(metric, value, baseline_value)
        self.save_metrics(current_metrics, metrics_output_path)

# Function to load test data
def load_test_data(data_path):
    logger.info("Loading test data from: %s", data_path)
    try:
        data = pd.read_csv(data_path)
        X_test = data.drop(columns=["target"])
        y_test = data["target"]
        return X_test, y_test
    except Exception as e:
        logger.error("Error loading test data: %s", str(e))
        raise

# Function to load threshold configuration from YAML
def load_threshold_config(config_path):
    logger.info("Loading threshold configuration from: %s", config_path)
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config["drift_thresholds"]
    except Exception as e:
        logger.error("Error loading configuration: %s", str(e))
        raise

# Main function
def main():
    MODEL_PATH = "/mnt/data/models/saved_models/best_model.pkl"
    BASELINE_METRICS_PATH = "/mnt/data/models/metrics/baseline_metrics.csv"
    TEST_DATA_PATH = "/mnt/data/data/processed/cleaned_customer_data.csv"
    METRICS_OUTPUT_PATH = "/mnt/data/models/metrics/latest_metrics.csv"

    logger.info("Starting model drift detection process")
    
    # Load test data
    X_test, y_test = load_test_data(TEST_DATA_PATH)
    
    # Load threshold configuration
    threshold_config = load_threshold_config(CONFIG_PATH)
    
    # Initialize the drift detector
    detector = ModelDriftDetector(
        model_path=MODEL_PATH,
        baseline_metrics_path=BASELINE_METRICS_PATH,
        threshold_config=threshold_config
    )
    
    # Monitor the model performance and detect drift
    detector.monitor(X_test, y_test, METRICS_OUTPUT_PATH)

if __name__ == "__main__":
    main()