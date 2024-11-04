import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetic parameters for plots
sns.set(style="whitegrid")
plt.style.use('ggplot')

# Load dataset
data_path = '../../data/processed/cleaned_customer_data.csv'
df = pd.read_csv(data_path)

# Basic data overview
print(f"Dataset shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")
print(f"Null values:\n{df.isnull().sum()}")

# Display the first few rows of the dataset
print(df.head())

# Convert any necessary columns to categorical types
df['Churn'] = df['Churn'].astype('category')

# Summary statistics of numeric columns
numeric_summary = df.describe()
print(f"Summary statistics:\n{numeric_summary}")

# Summary of categorical columns
categorical_summary = df.describe(include=['category'])
print(f"Categorical column summary:\n{categorical_summary}")

# Check for class imbalance in the target variable (Churn)
churn_distribution = df['Churn'].value_counts(normalize=True) * 100
print(f"Churn Distribution (percentage):\n{churn_distribution}")

# Visualization of churn distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title('Churn Distribution')
plt.ylabel('Count')
plt.xlabel('Churn')
plt.show()

# Distribution of numerical features
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Correlation heatmap of numerical features
plt.figure(figsize=(12, 8))
corr_matrix = df[numerical_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Boxplots to explore feature distribution by churn
for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y=col, data=df, palette='Set2')
    plt.title(f'{col} by Churn')
    plt.show()

# Pairplot for key numerical features
selected_columns = ['Tenure', 'MonthlyCharges', 'TotalCharges']
sns.pairplot(df[selected_columns + ['Churn']], hue='Churn', palette='Set1')
plt.show()

# Countplot for categorical features against churn
categorical_columns = df.select_dtypes(include=['category']).columns

for col in categorical_columns:
    if col != 'Churn':
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, hue='Churn', data=df, palette='Set3')
        plt.title(f'{col} Count by Churn')
        plt.ylabel('Count')
        plt.show()

# Visualizing correlation between categorical variables and churn using chi-square test of independence
from scipy.stats import chi2_contingency

def chi_square_test(df, col):
    contingency_table = pd.crosstab(df[col], df['Churn'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return p

# Apply the chi-square test to categorical columns
chi_square_results = {}
for col in categorical_columns:
    if col != 'Churn':
        p_value = chi_square_test(df, col)
        chi_square_results[col] = p_value

print(f"Chi-square test results (p-values):\n{chi_square_results}")

# Barplot of chi-square p-values for categorical variables
plt.figure(figsize=(10, 6))
sns.barplot(x=list(chi_square_results.keys()), y=list(chi_square_results.values()), palette='coolwarm')
plt.axhline(0.05, color='r', linestyle='--')  # Significance threshold
plt.title('Chi-square Test Results for Categorical Variables')
plt.ylabel('P-value')
plt.xlabel('Categorical Features')
plt.xticks(rotation=45)
plt.show()

# Correlation of churn with numeric variables
plt.figure(figsize=(10, 6))
df_corr = df.corr()['Churn'].sort_values(ascending=False)
df_corr.dropna(inplace=True)
sns.barplot(x=df_corr.index, y=df_corr.values, palette='viridis')
plt.title('Correlation with Churn')
plt.xticks(rotation=45)
plt.ylabel('Correlation')
plt.show()

# Visualize monthly charges distribution between churned and non-churned customers
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Churn'] == 1]['MonthlyCharges'], color='red', label='Churned', kde=True, bins=30)
sns.histplot(df[df['Churn'] == 0]['MonthlyCharges'], color='blue', label='Not Churned', kde=True, bins=30)
plt.title('Monthly Charges Distribution by Churn')
plt.legend()
plt.show()

# Visualize tenure distribution between churned and non-churned customers
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Churn'] == 1]['Tenure'], color='red', label='Churned', kde=True, bins=30)
sns.histplot(df[df['Churn'] == 0]['Tenure'], color='blue', label='Not Churned', kde=True, bins=30)
plt.title('Tenure Distribution by Churn')
plt.legend()
plt.show()