import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from features.feature_creation import create_features
from features.feature_selection import select_features
from features.scaling_normalization import scale_and_normalize

# Load data from CSV
@pytest.fixture(scope="module")
def sample_data():
    return pd.read_csv('features/feature_store/features.csv')

# Test feature creation
def test_create_features(sample_data):
    features = create_features(sample_data)
    assert 'age_tenure_ratio' in features.columns
    assert 'total_monthly_charges_ratio' in features.columns
    assert not features.isnull().values.any()

# Test feature selection based on correlation threshold
def test_feature_selection_corr(sample_data):
    selected_features = select_features(sample_data, method='correlation', threshold=0.8)
    assert len(selected_features.columns) > 0
    assert 'customer_id' not in selected_features.columns

# Test feature selection based on mutual information
def test_feature_selection_mi(sample_data):
    selected_features = select_features(sample_data, method='mutual_info', top_k=3)
    assert len(selected_features.columns) == 3
    assert 'age' in selected_features.columns

# Test scaling and normalization using StandardScaler
def test_standard_scaling(sample_data):
    numeric_columns = sample_data.select_dtypes(include=[np.number]).columns
    scaled_data = scale_and_normalize(sample_data[numeric_columns], StandardScaler())
    assert np.allclose(scaled_data.mean(), 0, atol=1e-7)
    assert np.allclose(scaled_data.std(), 1, atol=1e-7)

# Test scaling and normalization using MinMaxScaler
def test_minmax_scaling(sample_data):
    numeric_columns = sample_data.select_dtypes(include=[np.number]).columns
    scaled_data = scale_and_normalize(sample_data[numeric_columns], MinMaxScaler())
    assert scaled_data.min().min() == 0
    assert scaled_data.max().max() == 1

# Test for missing values in features
def test_handle_missing_values(sample_data):
    data_with_missing = sample_data.copy()
    data_with_missing.iloc[0, 1] = np.nan  # Introduce a missing value
    cleaned_data = create_features(data_with_missing.fillna(0))
    assert not cleaned_data.isnull().values.any()

# Test for handling outliers
def test_handle_outliers(sample_data):
    outlier_data = sample_data.copy()
    outlier_data.loc[0, 'monthly_charges'] = 10000  # Create an outlier
    scaled_data = scale_and_normalize(outlier_data, StandardScaler())
    assert scaled_data['monthly_charges'].mean() < 1e3

# Test for categorical feature encoding
def test_categorical_feature_encoding():
    data_with_categorical = pd.DataFrame({
        'gender': ['Male', 'Female', 'Female', 'Male', 'Male'],
        'senior_citizen': [0, 1, 0, 1, 0],
        'partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'churn': [0, 1, 1, 0, 1]
    })
    encoded_data = create_features(data_with_categorical)
    assert 'gender_Male' in encoded_data.columns
    assert 'gender_Female' in encoded_data.columns

# Test feature scaling without categorical variables
def test_scaling_without_categorical(sample_data):
    numeric_columns = sample_data.select_dtypes(include=[np.number]).columns
    scaled_data = scale_and_normalize(sample_data[numeric_columns], StandardScaler())
    assert np.allclose(scaled_data.mean(), 0, atol=1e-7)

# Test multi-threaded feature engineering
def test_multithreaded_feature_engineering(sample_data):
    from multiprocessing import Pool

    def process_chunk(chunk):
        return create_features(chunk)

    num_partitions = 2
    chunk_size = len(sample_data) // num_partitions
    chunks = [sample_data.iloc[i:i + chunk_size] for i in range(0, len(sample_data), chunk_size)]

    with Pool(num_partitions) as pool:
        results = pool.map(process_chunk, chunks)

    final_result = pd.concat(results)
    assert len(final_result) == len(sample_data)

# Test feature scaling with NaN values handling
def test_scaling_with_nan_handling(sample_data):
    data_with_nan = sample_data.copy()
    data_with_nan.iloc[0, 2] = np.nan  # Introduce a missing value
    data_with_nan_filled = data_with_nan.fillna(0)
    numeric_columns = data_with_nan_filled.select_dtypes(include=[np.number]).columns
    scaled_data = scale_and_normalize(data_with_nan_filled[numeric_columns], MinMaxScaler())
    assert scaled_data.isnull().sum().sum() == 0
    assert scaled_data.min().min() == 0

# Test feature selection using Lasso
def test_feature_selection_lasso(sample_data):
    selected_features = select_features(sample_data, method='lasso', alpha=0.01)
    assert 'age' in selected_features.columns

# Test feature creation from interaction terms
def test_interaction_terms(sample_data):
    interaction_features = create_features(sample_data, interaction_terms=True)
    assert 'age_tenure_interaction' in interaction_features.columns
    assert not interaction_features.isnull().values.any()

# Test normalization with very small values
def test_normalization_with_small_values():
    small_value_data = pd.DataFrame({
        'age': [0.00025, 0.00032, 0.00047, 0.00051, 0.00038],
        'tenure': [0.0005, 0.0010, 0.0003, 0.0007, 0.0002],
        'monthly_charges': [0.0002985, 0.0005695, 0.0005385, 0.0004230, 0.0007020]
    })
    scaled_data = scale_and_normalize(small_value_data, MinMaxScaler())
    assert np.allclose(scaled_data.mean(), 0.5, atol=1e-2)

# Test handling of large datasets
def test_large_dataset_handling():
    large_data = pd.DataFrame({
        'age': np.random.randint(18, 70, size=10000),
        'tenure': np.random.randint(1, 60, size=10000),
        'monthly_charges': np.random.uniform(20, 100, size=10000),
        'total_charges': np.random.uniform(100, 5000, size=10000)
    })
    scaled_data = scale_and_normalize(large_data, StandardScaler())
    assert scaled_data.shape == large_data.shape