import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Load the cleaned customer data
customer_data = pd.read_csv('../processed/cleaned_customer_data.csv')

# Load transaction data for feature generation
transaction_data = pd.read_csv('../raw/transaction_data.csv')

# Merge customer and transaction data
data = pd.merge(customer_data, transaction_data, on='customer_id', how='left')

# Handling missing values
def handle_missing_values(df):
    # Define strategy for numerical and categorical columns
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Separate numerical and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Impute missing values
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df

data = handle_missing_values(data)

# One-hot encoding for categorical features
def one_hot_encode(df, columns):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_df = pd.DataFrame(encoder.fit_transform(df[columns]))
    encoded_df.columns = encoder.get_feature_names_out(columns)
    df = df.drop(columns, axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df

categorical_columns = ['gender', 'country', 'product_type']
data = one_hot_encode(data, categorical_columns)

# Label encoding for target variable
def encode_target_variable(df, target_column):
    encoder = LabelEncoder()
    df[target_column] = encoder.fit_transform(df[target_column])
    return df

data = encode_target_variable(data, 'churn')

# Feature scaling (normalizing numerical features)
def scale_features(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

numerical_columns = ['age', 'income', 'transaction_amount', 'transaction_count']
data = scale_features(data, numerical_columns)

# Generate additional features from transaction data
def generate_transaction_features(df):
    # Create features from transaction data
    df['transaction_amount_per_count'] = df['transaction_amount'] / (df['transaction_count'] + 1)
    df['average_transaction'] = df.groupby('customer_id')['transaction_amount'].transform('mean')
    df['transaction_variance'] = df.groupby('customer_id')['transaction_amount'].transform('var')
    
    # Time-based features (with transaction timestamps)
    df['days_since_last_transaction'] = (pd.to_datetime('now') - pd.to_datetime(df['last_transaction_date'])).dt.days
    
    return df

data = generate_transaction_features(data)

# Principal Component Analysis (PCA) for dimensionality reduction
def apply_pca(df, n_components=5):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    pca_columns = [f'pca_{i}' for i in range(n_components)]
    pca_df = pd.DataFrame(principal_components, columns=pca_columns)
    return pca_df

pca_columns = ['transaction_amount', 'transaction_count', 'age', 'income']
pca_features = apply_pca(data[pca_columns])
data = pd.concat([data, pca_features], axis=1)

# Time-based features
def generate_time_features(df):
    df['year'] = pd.to_datetime(df['signup_date']).dt.year
    df['month'] = pd.to_datetime(df['signup_date']).dt.month
    df['day_of_week'] = pd.to_datetime(df['signup_date']).dt.weekday
    return df

data = generate_time_features(data)

# Aggregating historical customer data
def aggregate_customer_history(df):
    agg_features = df.groupby('customer_id').agg({
        'transaction_amount': ['mean', 'sum', 'min', 'max'],
        'transaction_count': ['mean', 'sum'],
        'days_since_last_transaction': ['mean', 'min', 'max']
    })
    
    # Flatten the MultiIndex columns
    agg_features.columns = ['_'.join(col).strip() for col in agg_features.columns.values]
    df = pd.merge(df, agg_features, on='customer_id', how='left')
    
    return df

data = aggregate_customer_history(data)

# Interaction terms
def generate_interaction_terms(df):
    df['age_income_interaction'] = df['age'] * df['income']
    df['transaction_amount_income_ratio'] = df['transaction_amount'] / (df['income'] + 1)
    return df

data = generate_interaction_terms(data)

# Drop redundant columns
def drop_redundant_columns(df, columns_to_drop):
    return df.drop(columns_to_drop, axis=1)

columns_to_drop = ['signup_date', 'last_transaction_date']
data = drop_redundant_columns(data, columns_to_drop)

# Save the final engineered dataset
data.to_csv('../processed/engineered_features.csv', index=False)

print("Feature engineering completed and saved.")