import optuna
from optuna import Trial
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
data = pd.read_csv('../data/processed/cleaned_customer_data.csv')
X = data.drop('churn', axis=1)
y = data['churn']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a function to optimize the RandomForestClassifier with Optuna
def objective_rf(trial: Trial):
    # Define hyperparameters for the RandomForestClassifier
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    # Define the classifier with trial parameters
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                bootstrap=bootstrap, random_state=42)

    # Use a pipeline to scale the data
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', rf)])
    
    pipeline.fit(X_train, y_train)
    
    # Predict on the validation set
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model using accuracy and F1-score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return f1

# Define a function to optimize XGBClassifier with Optuna
def objective_xgb(trial: Trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }
    
    xgb = XGBClassifier(**param, random_state=42, use_label_encoder=False)
    
    # Use a pipeline to scale the data
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', xgb)])
    
    pipeline.fit(X_train, y_train)
    
    # Predict on the validation set
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model using ROC-AUC
    roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    
    return roc_auc

# Experiment: tune RandomForestClassifier using Optuna
study_rf = optuna.create_study(direction='maximize')
logging.info("Starting RandomForest hyperparameter tuning with Optuna.")
study_rf.optimize(objective_rf, n_trials=100)

# Display best RandomForest parameters
logging.info(f"Best RandomForest params: {study_rf.best_params}")

# Experiment: tune XGBoost using Optuna
study_xgb = optuna.create_study(direction='maximize')
logging.info("Starting XGBoost hyperparameter tuning with Optuna.")
study_xgb.optimize(objective_xgb, n_trials=100)

# Display best XGBoost parameters
logging.info(f"Best XGBoost params: {study_xgb.best_params}")

# GridSearchCV for SVM 
logging.info("Starting GridSearchCV for SVM hyperparameter tuning.")
svm_parameters = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

svm = SVC(probability=True, random_state=42)
svm_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', svm)])

grid_search = GridSearchCV(estimator=svm_pipeline, param_grid=svm_parameters, 
                           scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

# Best SVM parameters
best_svm_params = grid_search.best_params_
logging.info(f"Best SVM params: {best_svm_params}")

# Final evaluation on the test set for all models
def evaluate_model(model_pipeline, model_name):
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model_pipeline.predict_proba(X_test)[:, 1])
    
    logging.info(f"{model_name} Evaluation - Accuracy: {accuracy}, F1-score: {f1}, ROC-AUC: {roc_auc}")

# Evaluate best models on test set
rf_best_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(**study_rf.best_params, random_state=42))])
xgb_best_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', XGBClassifier(**study_xgb.best_params, random_state=42, use_label_encoder=False))])
svm_best_pipeline = grid_search.best_estimator_

logging.info("Final evaluation on the test set for all models.")
evaluate_model(rf_best_pipeline, "RandomForest")
evaluate_model(xgb_best_pipeline, "XGBoost")
evaluate_model(svm_best_pipeline, "SVM")