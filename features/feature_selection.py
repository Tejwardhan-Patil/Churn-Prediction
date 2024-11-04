import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FeatureSelection')

# Load Data
def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    logger.info("Preprocessing data")
    
    # Handle missing values
    data = data.fillna(method='ffill')

    # Label encode categorical features
    label_encoder = LabelEncoder()
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # Feature scaling
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    return data

# Feature selection using RandomForest importance
def feature_importance_rf(X, y, num_features):
    logger.info("Selecting features using Random Forest importance")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = feature_importances.nlargest(num_features).index
    
    logger.info(f"Top {num_features} features selected: {top_features.tolist()}")
    return X[top_features]

# Feature selection using Recursive Feature Elimination (RFE)
def feature_selection_rfe(X, y, num_features):
    logger.info("Selecting features using Recursive Feature Elimination (RFE)")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=num_features)
    rfe.fit(X, y)

    selected_features = X.columns[rfe.support_]
    logger.info(f"Top {num_features} features selected by RFE: {selected_features.tolist()}")
    return X[selected_features]

# Feature selection using Mutual Information
def feature_selection_mi(X, y, num_features):
    logger.info("Selecting features using Mutual Information")

    mi = mutual_info_classif(X, y)
    mi_series = pd.Series(mi, index=X.columns)
    top_features = mi_series.nlargest(num_features).index

    logger.info(f"Top {num_features} features selected by Mutual Information: {top_features.tolist()}")
    return X[top_features]

# Feature selection using SelectKBest (Chi-square)
def feature_selection_chi2(X, y, num_features):
    logger.info("Selecting features using SelectKBest (Chi-square)")

    chi2_selector = SelectKBest(chi2, k=num_features)
    X_kbest = chi2_selector.fit_transform(X, y)
    
    selected_features = X.columns[chi2_selector.get_support()]
    logger.info(f"Top {num_features} features selected by Chi-square: {selected_features.tolist()}")
    return X[selected_features]

# Main function for feature selection pipeline
def feature_selection_pipeline(file_path, method='rf', num_features=10):
    data = load_data(file_path)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Preprocess data
    X = preprocess_data(X)

    # Choose feature selection method
    if method == 'rf':
        X_selected = feature_importance_rf(X, y, num_features)
    elif method == 'rfe':
        X_selected = feature_selection_rfe(X, y, num_features)
    elif method == 'mi':
        X_selected = feature_selection_mi(X, y, num_features)
    elif method == 'chi2':
        X_selected = feature_selection_chi2(X, y, num_features)
    else:
        raise ValueError("Invalid method provided for feature selection")

    logger.info(f"Final selected features: {X_selected.columns.tolist()}")
    return X_selected

# Function to split data after feature selection
def split_data(X, y, test_size=0.2, random_state=42):
    logger.info("Splitting data into training and test sets")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    # Path to the cleaned customer data
    data_path = "data/processed/cleaned_customer_data.csv"

    # Choose the method and number of features
    selected_method = 'rfe' 
    n_features = 10

    # Run the feature selection pipeline
    logger.info(f"Running feature selection with method: {selected_method} and number of features: {n_features}")
    
    selected_data = feature_selection_pipeline(data_path, method=selected_method, num_features=n_features)
    
    data = load_data(data_path)
    X, y = selected_data, data.iloc[:, -1]

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    logger.info(f"Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")