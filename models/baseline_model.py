import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# Paths for the dataset and saving the model
DATA_PATH = os.path.join(os.getcwd(), 'data/processed/cleaned_customer_data.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'models/saved_models/baseline_model.pkl')

# Load the cleaned customer data
def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)

# Preprocess the data
def preprocess_data(data):
    X = data.drop('churn', axis=1)
    y = data['churn']
    
    # Filling missing values with median
    X.fillna(X.median(), inplace=True)
    
    # Encoding for categorical features
    X = pd.get_dummies(X)
    
    return X, y

# Split the data into train, validation, and test sets
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Train the baseline logistic regression model
def train_baseline_model(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    
    print("Evaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

# Save the trained model
def save_model(model, save_path):
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

# Load the saved model
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

# Main function to orchestrate the process
def main():
    print("Loading data...")
    data = load_data(DATA_PATH)
    
    print("Preprocessing data...")
    X, y = preprocess_data(data)
    
    print("Splitting data into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    print("Training baseline model...")
    model = train_baseline_model(X_train, y_train)
    
    print("Evaluating model on validation set...")
    evaluate_model(model, X_val, y_val)
    
    print("Saving the trained model...")
    save_model(model, MODEL_SAVE_PATH)
    
    print("Loading the saved model...")
    loaded_model = load_model(MODEL_SAVE_PATH)
    
    print("Evaluating loaded model on test set...")
    evaluate_model(loaded_model, X_test, y_test)

if __name__ == "__main__":
    main()