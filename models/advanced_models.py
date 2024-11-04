import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Load and preprocess data
data = pd.read_csv('data/processed/cleaned_customer_data.csv')
X = data.drop(columns=['churn'])
y = data['churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- Random Forest --------------------
print("Training Random Forest Classifier...")

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1, verbose=2, scoring='f1')
rf_grid_search.fit(X_train, y_train)

best_rf = rf_grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Random Forest evaluation
print("Random Forest Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# -------------------- XGBoost --------------------
print("Training XGBoost Classifier...")

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False)
xgb_grid_search = GridSearchCV(xgb, xgb_params, cv=3, n_jobs=-1, verbose=2, scoring='f1')
xgb_grid_search.fit(X_train, y_train)

best_xgb = xgb_grid_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

# XGBoost evaluation
print("XGBoost Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# -------------------- Neural Network --------------------
print("Training Neural Network...")

mlp_params = {
    'hidden_layer_sizes': [(100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive']
}

mlp = MLPClassifier(random_state=42, max_iter=500)
mlp_grid_search = GridSearchCV(mlp, mlp_params, cv=3, n_jobs=-1, verbose=2, scoring='f1')
mlp_grid_search.fit(X_train_scaled, y_train)

best_mlp = mlp_grid_search.best_estimator_
y_pred_mlp = best_mlp.predict(X_test_scaled)

# Neural Network evaluation
print("Neural Network Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("F1 Score:", f1_score(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))

# -------------------- Model Comparison --------------------
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{name} - Accuracy: {acc}, F1 Score: {f1}")
    return acc, f1

print("Comparing Models:")
acc_rf, f1_rf = evaluate_model("Random Forest", y_test, y_pred_rf)
acc_xgb, f1_xgb = evaluate_model("XGBoost", y_test, y_pred_xgb)
acc_mlp, f1_mlp = evaluate_model("Neural Network", y_test, y_pred_mlp)

# Saving the best model based on F1 Score
best_model = None
if f1_rf >= f1_xgb and f1_rf >= f1_mlp:
    best_model = best_rf
    print("Best Model: Random Forest")
elif f1_xgb >= f1_rf and f1_xgb >= f1_mlp:
    best_model = best_xgb
    print("Best Model: XGBoost")
else:
    best_model = best_mlp
    print("Best Model: Neural Network")

# Saving the best model
import joblib
joblib.dump(best_model, 'models/saved_models/best_model.pkl')
print("Best model saved to models/saved_models/best_model.pkl")