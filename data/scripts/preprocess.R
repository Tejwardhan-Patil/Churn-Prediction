library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(lubridate)
library(caret)
library(ggplot2)

# Define file paths
raw_customer_data <- 'data/raw/customer_data.csv'
raw_transaction_data <- 'data/raw/transaction_data.csv'
processed_customer_data <- 'data/processed/cleaned_customer_data.csv'

# Load the raw data
customer_data <- read_csv(raw_customer_data)
transaction_data <- read_csv(raw_transaction_data)

# Data Cleaning - Handling missing values in customer data
missing_summary <- customer_data %>%
  summarize_all(~ sum(is.na(.)))

# Print missing value summary
print(missing_summary)

# Impute missing values for numerical columns using median
customer_data <- customer_data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Data Cleaning - Handling missing values in transaction data
missing_summary_transaction <- transaction_data %>%
  summarize_all(~ sum(is.na(.)))

print(missing_summary_transaction)

# Impute missing transaction values
transaction_data <- transaction_data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Remove duplicated rows
customer_data <- customer_data %>%
  distinct()

transaction_data <- transaction_data %>%
  distinct()

# Merge customer and transaction data based on customer ID
merged_data <- customer_data %>%
  left_join(transaction_data, by = "customer_id")

# Create new features

# Feature 1: Create a binary flag for high transaction value
merged_data <- merged_data %>%
  mutate(high_value_customer = if_else(total_transaction_amount > 5000, 1, 0))

# Feature 2: Convert date columns to Date format
merged_data <- merged_data %>%
  mutate(transaction_date = as.Date(transaction_date, format = "%Y-%m-%d"),
         signup_date = as.Date(signup_date, format = "%Y-%m-%d"))

# Feature 3: Calculate the difference between transaction date and signup date
merged_data <- merged_data %>%
  mutate(days_since_signup = as.numeric(difftime(transaction_date, signup_date, units = "days")))

# Handle categorical variables
merged_data <- merged_data %>%
  mutate(gender = factor(gender),
         region = factor(region),
         customer_segment = factor(customer_segment))

# One-hot encoding for categorical variables
merged_data <- merged_data %>%
  mutate_at(vars(gender, region, customer_segment), ~ as.numeric(as.factor(.)))

# Outlier detection and removal
transaction_threshold <- mean(merged_data$total_transaction_amount, na.rm = TRUE) +
  3 * sd(merged_data$total_transaction_amount, na.rm = TRUE)

# Remove outliers
merged_data <- merged_data %>%
  filter(total_transaction_amount <= transaction_threshold)

# Feature scaling and normalization
scale_features <- function(df, cols) {
  for (col in cols) {
    df[[col]] <- (df[[col]] - min(df[[col]], na.rm = TRUE)) / 
                 (max(df[[col]], na.rm = TRUE) - min(df[[col]], na.rm = TRUE))
  }
  return(df)
}

numeric_columns <- c("total_transaction_amount", "days_since_signup")
merged_data <- scale_features(merged_data, numeric_columns)

# Train-test split
set.seed(123)
trainIndex <- createDataPartition(merged_data$high_value_customer, p = .8, 
                                  list = FALSE, 
                                  times = 1)

train_data <- merged_data[trainIndex,]
test_data <- merged_data[-trainIndex,]

# Save the cleaned and processed data
write_csv(train_data, processed_customer_data)

# Exploratory Data Analysis (EDA) - Visualizing customer distribution
ggplot(merged_data, aes(x = high_value_customer)) +
  geom_bar() +
  ggtitle("Distribution of High Value Customers") +
  xlab("High Value Customer") +
  ylab("Count")

# EDA - Total transaction amount distribution
ggplot(merged_data, aes(x = total_transaction_amount)) +
  geom_histogram(bins = 50) +
  ggtitle("Distribution of Total Transaction Amount") +
  xlab("Total Transaction Amount") +
  ylab("Count")

# EDA - Days since signup vs total transaction amount
ggplot(merged_data, aes(x = days_since_signup, y = total_transaction_amount)) +
  geom_point() +
  ggtitle("Days Since Signup vs Total Transaction Amount") +
  xlab("Days Since Signup") +
  ylab("Total Transaction Amount")

# Data transformation for statistical models
processed_data <- merged_data %>%
  mutate(across(where(is.factor), as.numeric))

# Feature selection - Remove features with near-zero variance
nzv <- nearZeroVar(processed_data, saveMetrics = TRUE)
print(nzv)

processed_data <- processed_data[, !nzv$nzv]

# Save processed data for further analysis
final_processed_data <- 'data/processed/final_processed_data.csv'
write_csv(processed_data, final_processed_data)

# Summary of the final dataset
summary(processed_data)
str(processed_data)

# Data cleanup - Remove temporary objects from memory
rm(customer_data, transaction_data, merged_data, missing_summary, missing_summary_transaction)
gc()

print("Data preprocessing completed and saved to processed directory.")