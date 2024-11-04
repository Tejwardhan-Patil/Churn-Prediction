import os
import yaml
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.advanced_models import train_xgboost, train_random_forest, train_neural_network
from models.baseline_model import train_logistic_regression
from models.model_selection import select_best_model
from features.feature_creation import create_features
from monitoring.metrics.model_drift import check_model_drift
from monitoring.logging.log_config import setup_logging
from experiments.results import save_experiment_results

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = "../configs/experiment_01.yaml"
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Load raw data
raw_data_path = os.path.join(config['data']['raw_data_path'], 'customer_data.csv')
transaction_data_path = os.path.join(config['data']['raw_data_path'], 'transaction_data.csv')

customer_data = pd.read_csv(raw_data_path)
transaction_data = pd.read_csv(transaction_data_path)

# Preprocessing function
def preprocess_data(customer_data, transaction_data):
    logger.info("Starting data preprocessing...")
    
    # Merge customer and transaction data on customer_id
    merged_data = pd.merge(customer_data, transaction_data, on='customer_id')
    
    # Handling missing values
    merged_data.fillna(merged_data.median(), inplace=True)
    
    # Feature encoding: Convert categorical variables to numeric (using one-hot encoding)
    categorical_features = ['gender', 'country', 'product']
    merged_data = pd.get_dummies(merged_data, columns=categorical_features, drop_first=True)
    
    # Remove duplicates
    merged_data.drop_duplicates(subset='customer_id', inplace=True)
    
    # Drop irrelevant columns
    merged_data.drop(['signup_date', 'last_transaction_date'], axis=1, inplace=True)
    
    # Ensure no NaNs or infinite values remain
    merged_data.replace([pd.NA, pd.NaT, float('inf'), float('-inf')], 0, inplace=True)
    
    return merged_data

# Preprocess data
logger.info("Preprocessing data...")
data = preprocess_data(customer_data, transaction_data)

# Feature engineering
logger.info("Creating features...")
data = create_features(data)

# Split data into train, validation, and test sets
logger.info("Splitting data...")
train_data, test_data = train_test_split(data, test_size=config['data_split']['test_size'], random_state=config['random_seed'])
train_data, val_data = train_test_split(train_data, test_size=config['data_split']['validation_size'], random_state=config['random_seed'])

# Normalize features
scaler = StandardScaler()
logger.info("Normalizing data...")
X_train = scaler.fit_transform(train_data.drop('churn', axis=1))
X_val = scaler.transform(val_data.drop('churn', axis=1))
X_test = scaler.transform(test_data.drop('churn', axis=1))

y_train = train_data['churn']
y_val = val_data['churn']
y_test = test_data['churn']

# Dictionary to hold models and results
models = {}
results = {}

# Train Logistic Regression (Baseline)
if config['models']['logistic_regression']['enabled']:
    logger.info("Training Logistic Regression model...")
    logistic_regression_model = train_logistic_regression(X_train, y_train)
    models['logistic_regression'] = logistic_regression_model

# Train XGBoost
if config['models']['xgboost']['enabled']:
    logger.info("Training XGBoost model...")
    xgboost_model = train_xgboost(X_train, y_train, X_val, y_val)
    models['xgboost'] = xgboost_model

# Train Random Forest
if config['models']['random_forest']['enabled']:
    logger.info("Training Random Forest model...")
    random_forest_model = train_random_forest(X_train, y_train)
    models['random_forest'] = random_forest_model

# Train Neural Network
if config['models']['neural_network']['enabled']:
    logger.info("Training Neural Network model...")
    neural_network_model = train_neural_network(X_train, y_train, X_val, y_val)
    models['neural_network'] = neural_network_model

# Evaluate models
logger.info("Evaluating models...")
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    logger.info(f"Model {model_name}: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1-score={f1}")

# Select the best model
best_model_name = select_best_model(results)
best_model = models[best_model_name]
logger.info(f"Best model selected: {best_model_name}")

# Save the best model
model_save_path = os.path.join(config['models']['saved_models'], f"{best_model_name}_model.pkl")
logger.info(f"Saving the best model to {model_save_path}...")
best_model.save_model(model_save_path)

# Save experiment results
logger.info("Saving experiment results...")
experiment_results = pd.DataFrame(results).T
save_experiment_results(experiment_results, config['experiments']['results_path'])

# Check for model drift
logger.info("Checking for model drift...")
check_model_drift(config['monitoring']['drift_threshold'], X_test, y_test)

# Log the completion of the experiment
logger.info("Experiment completed successfully.")