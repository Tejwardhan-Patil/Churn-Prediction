library(dplyr)
library(tidyr)
library(lubridate)
library(caret)
library(ggplot2)
library(randomForest)
library(MASS)

# Load datasets
customer_data <- read.csv("../raw/customer_data.csv")
transaction_data <- read.csv("../raw/transaction_data.csv")

# Initial data merging (by customer_id)
data <- customer_data %>%
  left_join(transaction_data, by = "customer_id")

# Create new features from existing data
# 1. Time-based features
data$account_age <- as.numeric(difftime(Sys.Date(), as.Date(data$account_creation_date), units = "days"))
data$last_transaction_days <- as.numeric(difftime(Sys.Date(), as.Date(data$last_transaction_date), units = "days"))

# 2. Aggregated transaction features
transaction_summary <- transaction_data %>%
  group_by(customer_id) %>%
  summarise(
    total_transactions = n(),
    total_amount = sum(transaction_amount),
    average_transaction_value = mean(transaction_amount),
    max_transaction_value = max(transaction_amount),
    min_transaction_value = min(transaction_amount),
    std_transaction_value = sd(transaction_amount)
  )

# Merge transaction summary with customer data
data <- data %>%
  left_join(transaction_summary, by = "customer_id")

# 3. Customer interaction features
data$interaction_per_month <- data$total_transactions / (data$account_age / 30)

# Feature engineering for categorical variables
# Convert categorical variables to factors
data$gender <- as.factor(data$gender)
data$churn <- as.factor(data$churn)

# One-hot encoding for 'membership_level'
data <- data %>%
  mutate(
    membership_basic = ifelse(membership_level == "Basic", 1, 0),
    membership_premium = ifelse(membership_level == "Premium", 1, 0),
    membership_gold = ifelse(membership_level == "Gold", 1, 0)
  ) %>%
  select(-membership_level)

# 4. Interaction terms
data$interaction_total_transactions_account_age <- data$total_transactions * data$account_age

# 5. Polynomial features
data$account_age_squared <- data$account_age^2
data$total_transactions_squared <- data$total_transactions^2

# Handling missing values
# Replace missing values with median for numerical features
num_cols <- sapply(data, is.numeric)
data[num_cols] <- apply(data[num_cols], 2, function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

# Replace missing values for categorical variables with mode
mode_func <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
data$gender[is.na(data$gender)] <- mode_func(data$gender)

# Scaling and normalization
# Normalize numeric features using Min-Max scaling
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
data[num_cols] <- as.data.frame(lapply(data[num_cols], normalize))

# Feature selection using correlation
corr_matrix <- cor(data[num_cols], use = "complete.obs")
high_corr <- findCorrelation(corr_matrix, cutoff = 0.9)
data <- data %>% select(-all_of(high_corr))

# Random Forest feature importance
set.seed(123)
rf_model <- randomForest(churn ~ ., data = data, importance = TRUE)
importance <- as.data.frame(rf_model$importance)
important_features <- rownames(importance)[order(importance$MeanDecreaseGini, decreasing = TRUE)[1:10]]

# Filter dataset to retain only important features
data <- data %>% select(all_of(important_features), churn)

# Train-test split
set.seed(123)
trainIndex <- createDataPartition(data$churn, p = 0.8, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Save processed data
write.csv(train_data, "../processed/train_data.csv", row.names = FALSE)
write.csv(test_data, "../processed/test_data.csv", row.names = FALSE)

# Visualization of feature importance
ggplot(importance, aes(x = reorder(rownames(importance), MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Features", y = "Mean Decrease in Gini") +
  theme_minimal()

# Principal Component Analysis (PCA) for dimensionality reduction
pca_model <- prcomp(data[, -ncol(data)], center = TRUE, scale. = TRUE)
pca_data <- as.data.frame(pca_model$x)
pca_data$churn <- data$churn

# Explained variance
explained_variance <- summary(pca_model)$importance[2,]
cumulative_variance <- cumsum(explained_variance)
plot(cumulative_variance, type = "b", xlab = "Principal Components", ylab = "Cumulative Explained Variance")

# Save PCA results
write.csv(pca_data, "../processed/pca_data.csv", row.names = FALSE)

# K-means clustering to identify customer segments
set.seed(123)
kmeans_result <- kmeans(data[, -ncol(data)], centers = 3, nstart = 20)
data$cluster <- as.factor(kmeans_result$cluster)

# Visualizing clusters
ggplot(data, aes(x = account_age, y = total_transactions, color = cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "Customer Segments based on K-Means Clustering") +
  theme_minimal()

# Save final processed data with clusters
write.csv(data, "../processed/final_data_with_clusters.csv", row.names = FALSE)

# Save the R environment for further analysis
save.image(file = "../processed/feature_engineering.RData")