import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to input and output data directories
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

# Files
CUSTOMER_DATA_FILE = "customer_data.csv"
TRANSACTION_DATA_FILE = "transaction_data.csv"
CLEANED_CUSTOMER_FILE = "cleaned_customer_data.csv"
CLEANED_TRANSACTION_FILE = "cleaned_transaction_data.csv"

# Function to load datasets with error handling
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Empty data encountered in file: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise

# Function to clean data (handle missing values, outliers, duplicates)
def clean_data(df):
    initial_shape = df.shape
    df = df.drop_duplicates()  
    df.replace("?", np.nan, inplace=True)  
    df.fillna(method='ffill', inplace=True)  
    logging.info(f"Cleaned data. Original shape: {initial_shape}, Cleaned shape: {df.shape}")
    return df

# Function to encode categorical features with error handling
def encode_categorical(df, categorical_columns):
    label_encoders = {}
    for col in categorical_columns:
        try:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            logging.info(f"Encoded column: {col}")
        except Exception as e:
            logging.error(f"Error encoding column {col}: {str(e)}")
            raise
    return df, label_encoders

# Function to scale numerical features
def scale_numerical(df, numerical_columns):
    scaler = StandardScaler()
    try:
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        logging.info(f"Scaled numerical columns: {numerical_columns}")
    except Exception as e:
        logging.error(f"Error scaling numerical columns {numerical_columns}: {str(e)}")
        raise
    return df, scaler

# Function to preprocess customer data
def preprocess_customer_data(customer_data):
    customer_data = clean_data(customer_data)
    categorical_columns = ['Gender', 'Geography']  
    numerical_columns = ['Age', 'Balance', 'CreditScore', 'EstimatedSalary'] 

    customer_data, label_encoders = encode_categorical(customer_data, categorical_columns)
    customer_data, scaler = scale_numerical(customer_data, numerical_columns)
    
    return customer_data, label_encoders, scaler

# Function to preprocess transaction data
def preprocess_transaction_data(transaction_data):
    transaction_data = clean_data(transaction_data)
    return transaction_data

# Function to save cleaned data
def save_cleaned_data(df, file_name):
    output_path = os.path.join(PROCESSED_DATA_PATH, file_name)
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Saved cleaned data to {output_path}")
    except Exception as e:
        logging.error(f"Error saving file {output_path}: {str(e)}")
        raise

# Main preprocessing function
def main():
    logging.info("Preprocessing started")

    # Load raw customer and transaction data
    customer_data_path = os.path.join(RAW_DATA_PATH, CUSTOMER_DATA_FILE)
    transaction_data_path = os.path.join(RAW_DATA_PATH, TRANSACTION_DATA_FILE)

    customer_data = load_data(customer_data_path)
    transaction_data = load_data(transaction_data_path)

    # Preprocess customer and transaction data
    customer_data, label_encoders, scaler = preprocess_customer_data(customer_data)
    transaction_data = preprocess_transaction_data(transaction_data)

    # Save cleaned datasets
    save_cleaned_data(customer_data, CLEANED_CUSTOMER_FILE)
    save_cleaned_data(transaction_data, CLEANED_TRANSACTION_FILE)

    logging.info("Preprocessing complete")

# Entry point
if __name__ == "__main__":
    main()