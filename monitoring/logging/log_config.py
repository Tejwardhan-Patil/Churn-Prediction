import logging
import logging.config
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from prometheus_client import start_http_server, Counter, Summary

# Prometheus Metrics
REQUEST_COUNT = Counter('api_request_count', 'Total number of API requests')
ERROR_COUNT = Counter('api_error_count', 'Total number of API errors')
PREDICTION_TIME = Summary('prediction_latency_seconds', 'Time spent on model predictions')

# Start Prometheus server for metrics monitoring
def start_prometheus_server(port=8000):
    start_http_server(port)
    logging.info(f"Prometheus metrics server started on port {port}")

# Log Configuration
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE_PATH = os.path.join(LOG_DIR, 'app.log')

# Define logging config dict
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': LOG_FILE_PATH,
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 5,
            'encoding': 'utf8'
        },
        'error_file': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': os.path.join(LOG_DIR, 'error.log'),
            'when': 'midnight',
            'backupCount': 7,
            'encoding': 'utf8'
        }
    },
    'loggers': {
        '': {  # root logger
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'error_file']
        },
        'prometheus_client': {  # Suppress excessive Prometheus client logging
            'level': 'WARNING',
            'handlers': ['console']
        }
    }
}

# Apply logging configuration
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def log_api_request(endpoint: str):
    REQUEST_COUNT.inc()
    logger.info(f"API request to endpoint: {endpoint}")

def log_api_error(error_msg: str):
    ERROR_COUNT.inc()
    logger.error(f"API error occurred: {error_msg}")

@PREDICTION_TIME.time()
def log_prediction_time(model_name: str):
    logger.info(f"Model {model_name} prediction started")
    # Simulate model prediction time 
    import time
    time.sleep(1)
    logger.info(f"Model {model_name} prediction completed")

# Usage within an API route/application
def handle_request(endpoint: str):
    try:
        log_api_request(endpoint)
        # Simulate some processing 
        if endpoint == "/predict":
            log_prediction_time("churn_model_v1")
        elif endpoint == "/health":
            logger.info("Health check OK")
        else:
            raise ValueError("Unknown endpoint")
    except Exception as e:
        log_api_error(str(e))

if __name__ == "__main__":
    start_prometheus_server()

    # Simulate API requests
    handle_request("/predict")
    handle_request("/health")
    handle_request("/invalid_endpoint")