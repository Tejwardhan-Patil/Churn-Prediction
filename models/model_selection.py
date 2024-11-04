import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Load preprocessed data (features and labels)
data = pd.read_csv('data/processed/cleaned_customer_data.csv')
X = data.drop(columns=['churn'])
y = data['churn']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 30, 10), max_iter=500)
}

# Dictionary to store model performances
model_performance = {}

# Loop through models and evaluate performance
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    # Store performance metrics
    model_performance[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Convert performance metrics to DataFrame for comparison
performance_df = pd.DataFrame(model_performance).T

# Select the best model based on F1-score
best_model_name = performance_df['f1_score'].idxmax()
best_model = models[best_model_name]

print(f"Best model: {best_model_name} with F1-score: {performance_df.loc[best_model_name, 'f1_score']}")

# Save the best model
model_dir = 'models/saved_models/'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'best_model.pkl')
joblib.dump(best_model, model_path)

print(f"Best model saved at {model_path}")

# Output the performance metrics for all models
performance_df.to_csv('experiments/results/model_performance.csv', index=True)

# Loading and validating saved model
def load_best_model():
    loaded_model = joblib.load(model_path)
    val_pred = loaded_model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred)
    print(f"Loaded model F1-score on validation data: {val_f1}")

load_best_model()