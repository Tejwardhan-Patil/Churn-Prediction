import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.impute import SimpleImputer
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('scaling_normalization')

class FeatureScalerNormalizer:
    def __init__(self, scaling_strategy='standard', normalization=False):
        """
        Initialize the scaler and normalizer class with scaling strategy and normalization flag.

        Args:
            scaling_strategy (str): Type of scaling ('standard', 'minmax', 'robust').
            normalization (bool): If True, normalize features after scaling.
        """
        self.scaling_strategy = scaling_strategy
        self.normalization = normalization
        self.scaler = None
        self.normalizer = None
        self._initialize_scaler()

    def _initialize_scaler(self):
        """
        Initializes the scaler based on the chosen scaling strategy.
        """
        if self.scaling_strategy == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_strategy == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_strategy == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Invalid scaling strategy: {self.scaling_strategy}")
        
        if self.normalization:
            self.normalizer = Normalizer()

    def fit(self, X):
        """
        Fit the scaler and normalizer to the data.

        Args:
            X (pd.DataFrame or np.ndarray): The input data.
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Input data must be a pandas DataFrame or numpy array.")
        
        if self.scaler:
            self.scaler.fit(X)
        if self.normalizer:
            self.normalizer.fit(X)
        
        logger.info(f"Scaler {self.scaler} fitted to the data.")
        if self.normalizer:
            logger.info(f"Normalizer {self.normalizer} fitted to the data.")

    def transform(self, X):
        """
        Transform the data using the fitted scaler and normalizer.

        Args:
            X (pd.DataFrame or np.ndarray): The input data.
        
        Returns:
            Transformed data (pd.DataFrame or np.ndarray)
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Input data must be a pandas DataFrame or numpy array.")
        
        logger.info("Transforming data with scaler and normalizer...")
        X_scaled = self.scaler.transform(X) if self.scaler else X
        if self.normalizer:
            X_scaled = self.normalizer.transform(X_scaled)
        return X_scaled

    def fit_transform(self, X):
        """
        Fit the scaler/normalizer and then transform the data.

        Args:
            X (pd.DataFrame or np.ndarray): The input data.
        
        Returns:
            Transformed data (pd.DataFrame or np.ndarray)
        """
        self.fit(X)
        return self.transform(X)


def preprocess_data(df, features_to_scale, scaling_strategy='standard', normalization=False):
    """
    Preprocess data by scaling and normalizing selected features.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        features_to_scale (list): List of column names to scale and normalize.
        scaling_strategy (str): Scaling strategy to use ('standard', 'minmax', 'robust').
        normalization (bool): Whether to normalize the features after scaling.
    
    Returns:
        pd.DataFrame: DataFrame with scaled and normalized features.
    """
    logger.info("Starting data preprocessing...")
    
    # Handle missing values
    logger.info("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    df[features_to_scale] = imputer.fit_transform(df[features_to_scale])
    
    # Scaling and Normalizing
    logger.info(f"Scaling features with {scaling_strategy} strategy.")
    scaler_normalizer = FeatureScalerNormalizer(scaling_strategy, normalization)
    df[features_to_scale] = scaler_normalizer.fit_transform(df[features_to_scale])
    
    logger.info("Data preprocessing completed.")
    return df

def load_data(filepath):
    """
    Load dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    logger.info(f"Loading data from {filepath}...")
    try:
        data = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def save_data(df, filepath):
    """
    Save the processed DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): Processed DataFrame.
        filepath (str): Path to save the file.
    """
    logger.info(f"Saving data to {filepath}...")
    try:
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved successfully to {filepath}.")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise

def main():
    """
    Main function to execute the scaling and normalization process.
    """
    input_filepath = 'data/processed/cleaned_customer_data.csv'
    output_filepath = 'data/processed/scaled_normalized_customer_data.csv'
    features_to_scale = ['age', 'income', 'spending_score', 'account_balance']

    # Load data
    df = load_data(input_filepath)

    # Scale and normalize data
    df_scaled_normalized = preprocess_data(df, features_to_scale, scaling_strategy='minmax', normalization=True)

    # Save the transformed data
    save_data(df_scaled_normalized, output_filepath)

if __name__ == "__main__":
    main()