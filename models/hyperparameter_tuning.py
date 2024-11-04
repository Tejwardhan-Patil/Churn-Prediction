import optuna
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

# Load the dataset
data = pd.read_csv('data/processed/cleaned_customer_data.csv')
X = data.drop('churn', axis=1)
y = data['churn']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define objective function for Optuna
def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['random_forest', 'xgboost', 'mlp'])
    
    if model_type == 'random_forest':
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
    elif model_type == 'xgboost':
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        clf = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
    elif model_type == 'mlp':
        hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 10, 100)
        activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
        solver = trial.suggest_categorical('solver', ['sgd', 'adam'])
        alpha = trial.suggest_float('alpha', 1e-5, 1e-2)
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
        clf = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), activation=activation, solver=solver, alpha=alpha,
                            learning_rate=learning_rate, random_state=42, max_iter=1000)
    
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate the model using accuracy and F1 score
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return f1  # Optuna will optimize based on F1-score

# Optuna study to find the best hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Retrieve the best hyperparameters
best_params = study.best_params
print(f"Best parameters found by Optuna: {best_params}")

# Save the best model
best_model_type = best_params['model_type']
if best_model_type == 'random_forest':
    best_model = RandomForestClassifier(**{k: v for k, v in best_params.items() if k != 'model_type'}, random_state=42)
elif best_model_type == 'xgboost':
    best_model = XGBClassifier(**{k: v for k, v in best_params.items() if k != 'model_type'}, random_state=42)
else:
    best_model = MLPClassifier(**{k: v for k, v in best_params.items() if k != 'model_type'}, random_state=42)

best_model.fit(X_train_scaled, y_train)
joblib.dump(best_model, 'models/saved_models/best_model.pkl')

# GridSearchCV implementation for more exhaustive hyperparameter tuning 
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [6, 10, 15]
}

param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01]
}

# Choose model and param grid for exhaustive search
selected_model = 'random_forest' 
if selected_model == 'random_forest':
    model = RandomForestClassifier(random_state=42)
    param_grid = param_grid_rf
elif selected_model == 'xgboost':
    model = XGBClassifier(random_state=42)
    param_grid = param_grid_xgb
else:
    model = MLPClassifier(max_iter=1000, random_state=42)
    param_grid = param_grid_mlp

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Save the best estimator from GridSearchCV
best_grid_model = grid_search.best_estimator_
joblib.dump(best_grid_model, 'models/saved_models/best_grid_model.pkl')

# Evaluate the best model from GridSearchCV
y_pred_grid = best_grid_model.predict(X_test_scaled)
accuracy_grid = accuracy_score(y_test, y_pred_grid)
f1_grid = f1_score(y_test, y_pred_grid)
roc_auc_grid = roc_auc_score(y_test, best_grid_model.predict_proba(X_test_scaled)[:, 1])

print(f"Best model from GridSearchCV: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_grid}, F1-Score: {f1_grid}, ROC AUC: {roc_auc_grid}")