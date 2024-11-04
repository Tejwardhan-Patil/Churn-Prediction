import os
import luigi
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging


# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths for raw, processed, and split data
RAW_CUSTOMER_DATA_PATH = "data/raw/customer_data.csv"
RAW_TRANSACTION_DATA_PATH = "data/raw/transaction_data.csv"
PROCESSED_CUSTOMER_DATA_PATH = "data/processed/cleaned_customer_data.csv"
PROCESSED_TRANSACTION_DATA_PATH = "data/processed/cleaned_transaction_data.csv"
FEATURED_DATA_PATH = "features/feature_store/features.csv"
TRAIN_PATH = "data/processed/train.csv"
VALID_PATH = "data/processed/valid.csv"
TEST_PATH = "data/processed/test.csv"


# Loading raw customer data
class LoadCustomerData(luigi.Task):
    def output(self):
        return luigi.LocalTarget(RAW_CUSTOMER_DATA_PATH)

    def run(self):
        logger.info("Loading customer data...")
        # Simulate loading data
        customer_data = {
            "customer_id": [1, 2, 3],
            "age": [34, 25, 45],
            "churn": [0, 1, 0]
        }
        df = pd.DataFrame(customer_data)
        df.to_csv(self.output().path, index=False)
        logger.info("Customer data loaded and saved.")


# Loading raw transaction data
class LoadTransactionData(luigi.Task):
    def output(self):
        return luigi.LocalTarget(RAW_TRANSACTION_DATA_PATH)

    def run(self):
        logger.info("Loading transaction data...")
        transaction_data = {
            "transaction_id": [101, 102, 103],
            "customer_id": [1, 2, 3],
            "amount": [250, 150, 300]
        }
        df = pd.DataFrame(transaction_data)
        df.to_csv(self.output().path, index=False)
        logger.info("Transaction data loaded and saved.")


# Cleaning customer data
class CleanCustomerData(luigi.Task):
    def requires(self):
        return LoadCustomerData()

    def output(self):
        return luigi.LocalTarget(PROCESSED_CUSTOMER_DATA_PATH)

    def run(self):
        logger.info("Cleaning customer data...")
        customer_data = pd.read_csv(self.input().path)
        # Perform some data cleaning (handling missing values)
        cleaned_data = customer_data.dropna()
        cleaned_data.to_csv(self.output().path, index=False)
        logger.info("Customer data cleaned and saved.")


# Cleaning transaction data
class CleanTransactionData(luigi.Task):
    def requires(self):
        return LoadTransactionData()

    def output(self):
        return luigi.LocalTarget(PROCESSED_TRANSACTION_DATA_PATH)

    def run(self):
        logger.info("Cleaning transaction data...")
        transaction_data = pd.read_csv(self.input().path)
        # Perform some data cleaning (removing outliers)
        cleaned_data = transaction_data[transaction_data['amount'] > 0]
        cleaned_data.to_csv(self.output().path, index=False)
        logger.info("Transaction data cleaned and saved.")


# Feature engineering
class FeatureEngineering(luigi.Task):
    def requires(self):
        return {
            "customer_data": CleanCustomerData(),
            "transaction_data": CleanTransactionData()
        }

    def output(self):
        return luigi.LocalTarget(FEATURED_DATA_PATH)

    def run(self):
        logger.info("Performing feature engineering...")
        customer_data = pd.read_csv(self.input()["customer_data"].path)
        transaction_data = pd.read_csv(self.input()["transaction_data"].path)

        # Join dataframes on customer_id
        merged_data = pd.merge(customer_data, transaction_data, on="customer_id")

        # Feature engineering: calculate total transactions
        merged_data["total_amount"] = merged_data.groupby("customer_id")["amount"].transform("sum")

        # Save engineered features
        merged_data.to_csv(self.output().path, index=False)
        logger.info("Feature engineering completed and saved.")


# Splitting the data into train, validation, and test sets
class DataSplit(luigi.Task):
    test_size = 0.2
    valid_size = 0.1

    def requires(self):
        return FeatureEngineering()

    def output(self):
        return {
            "train": luigi.LocalTarget(TRAIN_PATH),
            "valid": luigi.LocalTarget(VALID_PATH),
            "test": luigi.LocalTarget(TEST_PATH)
        }

    def run(self):
        logger.info("Splitting data into train, validation, and test sets...")
        data = pd.read_csv(self.input().path)

        # Splitting the data
        train, temp = train_test_split(data, test_size=self.test_size + self.valid_size)
        valid, test = train_test_split(temp, test_size=self.test_size / (self.test_size + self.valid_size))

        # Save the splits
        train.to_csv(self.output()["train"].path, index=False)
        valid.to_csv(self.output()["valid"].path, index=False)
        test.to_csv(self.output()["test"].path, index=False)
        logger.info("Data split and saved.")


# Scaling and normalization
class ScalingNormalization(luigi.Task):
    def requires(self):
        return DataSplit()

    def output(self):
        return {
            "scaled_train": luigi.LocalTarget("data/processed/scaled_train.csv"),
            "scaled_valid": luigi.LocalTarget("data/processed/scaled_valid.csv"),
            "scaled_test": luigi.LocalTarget("data/processed/scaled_test.csv")
        }

    def run(self):
        logger.info("Scaling and normalizing data...")
        scaler = StandardScaler()

        # Load datasets
        train = pd.read_csv(self.input()["train"].path)
        valid = pd.read_csv(self.input()["valid"].path)
        test = pd.read_csv(self.input()["test"].path)

        # Scale features
        scaled_train = scaler.fit_transform(train)
        scaled_valid = scaler.transform(valid)
        scaled_test = scaler.transform(test)

        # Save scaled data
        pd.DataFrame(scaled_train).to_csv(self.output()["scaled_train"].path, index=False)
        pd.DataFrame(scaled_valid).to_csv(self.output()["scaled_valid"].path, index=False)
        pd.DataFrame(scaled_test).to_csv(self.output()["scaled_test"].path, index=False)
        logger.info("Data scaling and normalization completed.")


# Running the entire pipeline
class ChurnPredictionPipeline(luigi.WrapperTask):
    def requires(self):
        return ScalingNormalization()


if __name__ == '__main__':
    luigi.run()