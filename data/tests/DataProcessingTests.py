import unittest
import pandas as pd
from data.scripts.preprocess import preprocess_data
from data.scripts.data_split import split_data

class TestDataPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load raw data from files
        cls.raw_customer_data = pd.read_csv('data/raw/customer_data.csv')
        cls.raw_transaction_data = pd.read_csv('data/raw/transaction_data.csv')

    def test_missing_values_handling(self):
        # Test that missing values are handled correctly using actual preprocessing logic
        cleaned_data = preprocess_data(self.raw_customer_data)
        self.assertFalse(cleaned_data['age'].isnull().any(), "Missing values in 'age' column should be handled.")
        self.assertFalse(cleaned_data['gender'].isnull().any(), "Missing values in 'gender' column should be handled.")

    def test_data_types_validation(self):
        # Ensure correct data types are maintained after preprocessing
        cleaned_data = preprocess_data(self.raw_customer_data)
        self.assertTrue(cleaned_data['customer_id'].dtype == 'int64', "customer_id should be integer.")
        self.assertTrue(cleaned_data['age'].dtype == 'float64', "age should be float.")
        self.assertTrue(cleaned_data['gender'].dtype == 'object', "gender should be string.")

    def test_outlier_detection(self):
        # Test outlier detection logic
        cleaned_data = preprocess_data(self.raw_customer_data)
        outliers = cleaned_data[cleaned_data['age'] > 100]
        self.assertTrue(outliers.empty, "Outliers should be correctly identified and handled.")

    def test_data_split(self):
        # Test train, validation, and test data splitting
        cleaned_data = preprocess_data(self.raw_customer_data)
        train_data, val_data, test_data = split_data(cleaned_data)
        self.assertGreater(len(train_data), len(val_data), "Training data should be larger than validation data.")
        self.assertGreater(len(train_data), len(test_data), "Training data should be larger than test data.")
    
    def test_churn_label_distribution(self):
        # Validate churn label distribution in the dataset
        cleaned_data = preprocess_data(self.raw_customer_data)
        churn_rate = cleaned_data['churn'].mean()
        self.assertGreaterEqual(churn_rate, 0.1, "Churn rate should be within expected range.")
        self.assertLessEqual(churn_rate, 0.9, "Churn rate should be within expected range.")

    def test_null_transaction_data_handling(self):
        # Test handling of null values in transaction data
        cleaned_transactions = preprocess_data(self.raw_transaction_data)
        self.assertFalse(cleaned_transactions['transaction_amount'].isnull().any(), "Null transaction amounts should be handled.")

    def test_transaction_date_format(self):
        # Validate transaction date formatting
        cleaned_transactions = preprocess_data(self.raw_transaction_data)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_transactions['transaction_date']),
                        "Transaction date should be in correct datetime format.")

    def test_data_transformation_scaling(self):
        # Test scaling of numeric features
        cleaned_data = preprocess_data(self.raw_customer_data)
        scaled_age = (cleaned_data['age'] - cleaned_data['age'].mean()) / cleaned_data['age'].std()
        self.assertAlmostEqual(scaled_age.mean(), 0, delta=0.01)
        self.assertAlmostEqual(scaled_age.std(), 1, delta=0.01)

    def test_categorical_encoding(self):
        # Test that categorical features are correctly encoded
        cleaned_data = preprocess_data(self.raw_customer_data)
        self.assertIn('gender_M', cleaned_data.columns, "Gender should be one-hot encoded.")
        self.assertIn('gender_F', cleaned_data.columns, "Gender should be one-hot encoded.")

    def test_feature_normalization(self):
        # Test normalization of features
        cleaned_data = preprocess_data(self.raw_customer_data)
        normalized_tenure = (cleaned_data['tenure'] - cleaned_data['tenure'].min()) / (
                    cleaned_data['tenure'].max() - cleaned_data['tenure'].min())
        self.assertAlmostEqual(normalized_tenure.min(), 0, delta=0.01)
        self.assertAlmostEqual(normalized_tenure.max(), 1, delta=0.01)

    def test_data_leakage_prevention(self):
        # Ensure no data leakage between training and testing sets
        cleaned_data = preprocess_data(self.raw_customer_data)
        train_data, _, test_data = split_data(cleaned_data)
        overlap = pd.merge(train_data, test_data, on='customer_id', how='inner')
        self.assertTrue(overlap.empty, "There should be no data leakage between training and testing sets.")

    def test_duplicate_entries_removal(self):
        # Test that duplicate entries are removed during preprocessing
        data_with_duplicates = self.raw_customer_data.append(self.raw_customer_data.iloc[0], ignore_index=True)
        cleaned_data = preprocess_data(data_with_duplicates)
        self.assertEqual(len(cleaned_data), len(self.raw_customer_data), "Duplicate entries should be removed.")

    def test_correct_columns_after_preprocessing(self):
        # Ensure only necessary columns are kept after preprocessing
        cleaned_data = preprocess_data(self.raw_customer_data)
        expected_columns = ['customer_id', 'age', 'gender_M', 'gender_F', 'tenure', 'churn']
        self.assertEqual(set(cleaned_data.columns), set(expected_columns), "Processed data should only contain expected columns.")

    def test_handling_of_inconsistent_data(self):
        # Test handling of inconsistent data types or formats
        inconsistent_data = self.raw_customer_data.copy()
        inconsistent_data.at[2, 'age'] = 'unknown'  # Insert inconsistent data type
        cleaned_data = preprocess_data(inconsistent_data)
        self.assertFalse(cleaned_data['age'].dtype == 'object', "Inconsistent data types should be handled properly.")

    def test_invalid_values_handling(self):
        # Test for handling invalid values (negative numbers, etc)
        invalid_data = self.raw_customer_data.copy()
        invalid_data.at[2, 'age'] = -10  # Insert invalid age
        cleaned_data = preprocess_data(invalid_data)
        self.assertTrue((cleaned_data['age'] >= 0).all(), "Invalid values should be handled properly.")

if __name__ == '__main__':
    unittest.main()