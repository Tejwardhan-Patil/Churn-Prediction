import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Define file paths
DATA_DIR = Path('data/processed')
RAW_DATA_FILE = DATA_DIR / 'cleaned_customer_data.csv'
TRAIN_FILE = DATA_DIR / 'train_data.csv'
VALIDATION_FILE = DATA_DIR / 'validation_data.csv'
TEST_FILE = DATA_DIR / 'test_data.csv'

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned customer data
def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} does not exist.")
    return pd.read_csv(file_path)

# Split data into train, validation, and test sets
def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)
    return train_df, val_df, test_df

# Save the split datasets to CSV files
def save_split_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VALIDATION_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    print(f"Training data saved to {TRAIN_FILE}")
    print(f"Validation data saved to {VALIDATION_FILE}")
    print(f"Test data saved to {TEST_FILE}")

# Main function to handle the data splitting
def main():
    # Load data
    print(f"Loading data from {RAW_DATA_FILE}...")
    df = load_data(RAW_DATA_FILE)
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Splitting the data
    print("Splitting data into train, validation, and test sets...")
    train_df, val_df, test_df = split_data(df)
    print(f"Training set: {train_df.shape[0]} samples")
    print(f"Validation set: {val_df.shape[0]} samples")
    print(f"Test set: {test_df.shape[0]} samples")

    # Save split data
    save_split_data(train_df, val_df, test_df)

# Define utility functions for additional functionality
def get_summary_statistics(df: pd.DataFrame):
    """
    Function to get summary statistics for each column in the dataset
    """
    return df.describe()

def check_missing_values(df: pd.DataFrame):
    """
    Function to check for missing values in the dataset
    """
    return df.isnull().sum()

def print_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Function to print a summary of the data split
    """
    print("\nData Split Summary:")
    print(f"Training Set: {train_df.shape[0]} rows")
    print(f"Validation Set: {val_df.shape[0]} rows")
    print(f"Test Set: {test_df.shape[0]} rows")

# Add class for handling complex preprocessing before the split 
class Preprocessing:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def remove_outliers(self):
        """
        Function for outlier removal based on z-scores
        """
        from scipy import stats
        z_scores = stats.zscore(self.df.select_dtypes(include=['float', 'int']))
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        self.df = self.df[filtered_entries]
        return self.df

    def encode_categorical(self):
        """
        Function to encode categorical variables using one-hot encoding
        """
        self.df = pd.get_dummies(self.df, drop_first=True)
        return self.df

    def fill_missing(self, strategy: str = 'mean'):
        """
        Function to fill missing values
        """
        if strategy == 'mean':
            self.df.fillna(self.df.mean(), inplace=True)
        elif strategy == 'median':
            self.df.fillna(self.df.median(), inplace=True)
        elif strategy == 'mode':
            self.df.fillna(self.df.mode().iloc[0], inplace=True)
        return self.df

# Create a function for more advanced data splitting based on specific columns
def stratified_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """
    Perform stratified split on the dataset based on the target column.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    for train_idx, test_idx in strat_split.split(df, df[target_col]):
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]
    
    val_split = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    
    for train_idx, val_idx in val_split.split(train_df, train_df[target_col]):
        train_df = train_df.loc[train_idx]
        val_df = train_df.loc[val_idx]
    
    return train_df, val_df, test_df

# Enhanced main function to handle preprocessing and advanced split
def enhanced_main():
    # Load and preprocess data
    df = load_data(RAW_DATA_FILE)
    
    preprocessor = Preprocessing(df)
    df = preprocessor.fill_missing(strategy='mean')
    df = preprocessor.remove_outliers()
    df = preprocessor.encode_categorical()

    # Perform stratified split
    print("Performing stratified split based on target column...")
    train_df, val_df, test_df = stratified_split(df, target_col='Churn')
    
    # Save datasets
    save_split_data(train_df, val_df, test_df)
    print_split_summary(train_df, val_df, test_df)

if __name__ == "__main__":
    enhanced_main()