import time
import psutil
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from prometheus_client import Gauge, Counter, start_http_server

# Setting up logging
logging.basicConfig(filename='performance_tracking.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prometheus metrics
accuracy_gauge = Gauge('model_accuracy', 'Accuracy of the model')
precision_gauge = Gauge('model_precision', 'Precision of the model')
recall_gauge = Gauge('model_recall', 'Recall of the model')
f1_gauge = Gauge('model_f1_score', 'F1 Score of the model')
auc_roc_gauge = Gauge('model_auc_roc', 'AUC-ROC of the model')
drift_counter = Counter('model_drift_detected', 'Number of times model drift was detected')
latency_gauge = Gauge('model_latency', 'Latency of model predictions')

# Start Prometheus server on port 8000
start_http_server(8000)

class ModelPerformanceTracker:
    def __init__(self, model):
        self.model = model
        self.prediction_times = []

    def predict(self, X):
        start_time = time.time()
        predictions = self.model.predict(X)
        end_time = time.time()
        latency = end_time - start_time
        self.prediction_times.append(latency)
        latency_gauge.set(latency)
        logging.info(f'Prediction latency: {latency:.4f} seconds')
        return predictions

    def track_metrics(self, y_true, y_pred):
        # Calculate various performance metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_pred)

        # Logging and updating Prometheus gauges
        logging.info(f'Accuracy: {accuracy:.4f}')
        logging.info(f'Precision: {precision:.4f}')
        logging.info(f'Recall: {recall:.4f}')
        logging.info(f'F1 Score: {f1:.4f}')
        logging.info(f'AUC-ROC: {auc_roc:.4f}')

        accuracy_gauge.set(accuracy)
        precision_gauge.set(precision)
        recall_gauge.set(recall)
        f1_gauge.set(f1)
        auc_roc_gauge.set(auc_roc)

    def detect_drift(self, X_train, y_train, X_test, y_test, threshold=0.05):
        # Detect model drift by comparing performance on training and test sets
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        drift_score = np.abs(train_accuracy - test_accuracy)
        if drift_score > threshold:
            logging.warning(f'Model drift detected. Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
            drift_counter.inc()

    def calculate_average_latency(self):
        avg_latency = np.mean(self.prediction_times)
        logging.info(f'Average prediction latency: {avg_latency:.4f} seconds')
        return avg_latency

    def system_monitoring(self):
        # Monitoring system resource usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        logging.info(f'CPU usage: {cpu_usage}%')
        logging.info(f'Memory usage: {memory_usage}%')

        return cpu_usage, memory_usage

# Usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate synthetic dataset for tracking performance
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Initialize the performance tracker
    performance_tracker = ModelPerformanceTracker(model)

    # Perform predictions and track metrics
    y_pred = performance_tracker.predict(X_test)
    performance_tracker.track_metrics(y_test, y_pred)

    # Check for model drift
    performance_tracker.detect_drift(X_train, y_train, X_test, y_test)

    # Log system usage
    cpu, memory = performance_tracker.system_monitoring()

    # Calculate and log average latency
    avg_latency = performance_tracker.calculate_average_latency()

    # Output summary
    print(f"Performance tracked and metrics are logged in performance_tracking.log.")
    print(f"Prometheus metrics are exposed on port 8000.")