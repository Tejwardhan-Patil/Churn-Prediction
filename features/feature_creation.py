import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load customer data and transaction data
customer_data = pd.read_csv('data/raw/customer_data.csv')
transaction_data = pd.read_csv('data/raw/transaction_data.csv')

# Function to create age from birth date
def create_age_feature(df):
    df['birth_date'] = pd.to_datetime(df['birth_date'])
    current_year = pd.Timestamp.now().year
    df['age'] = current_year - df['birth_date'].dt.year
    return df

# Function to create total purchase amount feature
def create_total_purchase(df):
    df['total_purchase'] = df['purchase_amount'].sum(axis=1)
    return df

# Function to create average purchase amount feature
def create_avg_purchase(df):
    df['avg_purchase'] = df['purchase_amount'].mean(axis=1)
    return df

# Function to create count of transactions feature
def create_transaction_count(df):
    transaction_count = df.groupby('customer_id').size().reset_index(name='transaction_count')
    return transaction_count

# Function to create recency feature (days since last purchase)
def create_recency_feature(df):
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    most_recent = df.groupby('customer_id')['transaction_date'].max().reset_index()
    most_recent['recency'] = (pd.Timestamp.now() - most_recent['transaction_date']).dt.days
    return most_recent[['customer_id', 'recency']]

# One-hot encoding for categorical variables
def encode_categorical_features(df, categorical_cols):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
    encoded_features.columns = encoder.get_feature_names(categorical_cols)
    return pd.concat([df.drop(columns=categorical_cols), encoded_features], axis=1)

# Label encoding for binary features
def encode_binary_features(df, binary_cols):
    encoder = LabelEncoder()
    for col in binary_cols:
        df[col] = encoder.fit_transform(df[col])
    return df

# Imputation for missing values
def impute_missing_values(df, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    imputed_data = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return imputed_data

# Function to extract text features using TF-IDF
def extract_text_features(df, text_col):
    vectorizer = TfidfVectorizer(max_features=50)
    tfidf_matrix = vectorizer.fit_transform(df[text_col])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return pd.concat([df.drop(columns=[text_col]), tfidf_df], axis=1)

# Merging customer and transaction data
def merge_customer_transaction_data(customer_df, transaction_df):
    return pd.merge(customer_df, transaction_df, on='customer_id', how='left')

# Feature creation pipeline
def create_features(customer_df, transaction_df):
    # Creating age feature
    customer_df = create_age_feature(customer_df)

    # Creating transaction-related features
    transaction_df = create_total_purchase(transaction_df)
    transaction_df = create_avg_purchase(transaction_df)
    transaction_count = create_transaction_count(transaction_df)
    recency = create_recency_feature(transaction_df)

    # Merging transaction features with customer data
    customer_features = merge_customer_transaction_data(customer_df, transaction_count)
    customer_features = merge_customer_transaction_data(customer_features, recency)

    # Encoding categorical features
    categorical_cols = ['gender', 'region']
    customer_features = encode_categorical_features(customer_features, categorical_cols)

    # Encoding binary features
    binary_cols = ['is_active', 'is_churned']
    customer_features = encode_binary_features(customer_features, binary_cols)

    # Imputing missing values
    customer_features = impute_missing_values(customer_features)

    # Extracting text features from customer reviews or feedback
    if 'feedback' in customer_features.columns:
        customer_features = extract_text_features(customer_features, 'feedback')

    return customer_features

# Saving the created features to CSV
def save_features_to_csv(features_df, output_path):
    features_df.to_csv(output_path, index=False)

# Load data
customer_data = pd.read_csv('data/raw/customer_data.csv')
transaction_data = pd.read_csv('data/raw/transaction_data.csv')

# Create features
features = create_features(customer_data, transaction_data)

# Save the features
save_features_to_csv(features, 'features/feature_store/features.csv')

print("Feature creation completed and saved to feature store.")