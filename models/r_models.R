library(glmnet)
library(randomForest)
library(xgboost)
library(caret)
library(tidyverse)
library(pROC)

# Set seed for reproducibility
set.seed(42)

# Load dataset
data <- read.csv('data/processed/cleaned_customer_data.csv')

# Data preprocessing
data <- data %>%
  mutate(Churn = factor(Churn, levels = c(0, 1))) %>%
  select(-CustomerID) # Dropping ID column

# Split data into training and testing sets
train_index <- createDataPartition(data$Churn, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Separate features and labels for training and testing
x_train <- model.matrix(Churn ~ ., data = train_data)[, -1]
y_train <- train_data$Churn
x_test <- model.matrix(Churn ~ ., data = test_data)[, -1]
y_test <- test_data$Churn

# GLMNET Model (Lasso and Ridge Regression)
cv_glmnet <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5, nfolds = 10)
best_lambda <- cv_glmnet$lambda.min

# Train final GLMNET model
glmnet_model <- glmnet(x_train, y_train, family = "binomial", lambda = best_lambda, alpha = 0.5)

# Predictions and evaluation for GLMNET model
glmnet_probs <- predict(glmnet_model, newx = x_test, type = "response")
glmnet_pred <- ifelse(glmnet_probs > 0.5, 1, 0)

# Calculate accuracy, AUC, and confusion matrix for GLMNET
glmnet_acc <- mean(glmnet_pred == y_test)
glmnet_roc <- roc(y_test, glmnet_probs)
glmnet_auc <- auc(glmnet_roc)
glmnet_confusion <- confusionMatrix(as.factor(glmnet_pred), y_test)

# Random Forest Model
rf_model <- randomForest(Churn ~ ., data = train_data, ntree = 500, mtry = 3, importance = TRUE)

# Predictions and evaluation for Random Forest model
rf_pred <- predict(rf_model, newdata = test_data)
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[, 2]

# Calculate accuracy, AUC, and confusion matrix for Random Forest
rf_acc <- mean(rf_pred == y_test)
rf_roc <- roc(y_test, rf_probs)
rf_auc <- auc(rf_roc)
rf_confusion <- confusionMatrix(as.factor(rf_pred), y_test)

# XGBoost Model
dtrain <- xgb.DMatrix(data = x_train, label = as.numeric(y_train) - 1)
dtest <- xgb.DMatrix(data = x_test, label = as.numeric(y_test) - 1)

# Define parameters for XGBoost
xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Cross-validation for XGBoost
xgb_cv <- xgb.cv(
  params = xgb_params,
  data = dtrain,
  nfold = 10,
  nrounds = 100,
  early_stopping_rounds = 10,
  verbose = 0
)

best_nrounds <- xgb_cv$best_iteration

# Train XGBoost model
xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = best_nrounds
)

# Predictions and evaluation for XGBoost model
xgb_probs <- predict(xgb_model, newdata = dtest)
xgb_pred <- ifelse(xgb_probs > 0.5, 1, 0)

# Calculate accuracy, AUC, and confusion matrix for XGBoost
xgb_acc <- mean(xgb_pred == y_test)
xgb_roc <- roc(y_test, xgb_probs)
xgb_auc <- auc(xgb_roc)
xgb_confusion <- confusionMatrix(as.factor(xgb_pred), y_test)

# Print performance metrics
cat("Performance Metrics\n")
cat("===================\n")
cat("GLMNET Model:\n")
cat("  Accuracy:", glmnet_acc, "\n")
cat("  AUC:", glmnet_auc, "\n")
cat("  Confusion Matrix:\n")
print(glmnet_confusion)

cat("\nRandom Forest Model:\n")
cat("  Accuracy:", rf_acc, "\n")
cat("  AUC:", rf_auc, "\n")
cat("  Confusion Matrix:\n")
print(rf_confusion)

cat("\nXGBoost Model:\n")
cat("  Accuracy:", xgb_acc, "\n")
cat("  AUC:", xgb_auc, "\n")
cat("  Confusion Matrix:\n")
print(xgb_confusion)

# Model Comparison
performance_df <- data.frame(
  Model = c("GLMNET", "Random Forest", "XGBoost"),
  Accuracy = c(glmnet_acc, rf_acc, xgb_acc),
  AUC = c(glmnet_auc, rf_auc, xgb_auc)
)

cat("\nModel Comparison:\n")
print(performance_df)

# Save best model (based on AUC)
best_model <- ifelse(performance_df$AUC[which.max(performance_df$AUC)] == glmnet_auc, "GLMNET", 
                     ifelse(performance_df$AUC[which.max(performance_df$AUC)] == rf_auc, "Random Forest", 
                            "XGBoost"))

cat("\nBest Model Selected:", best_model, "\n")

# Save the best model
if (best_model == "GLMNET") {
  saveRDS(glmnet_model, file = "models/saved_models/glmnet_best_model.rds")
} else if (best_model == "Random Forest") {
  saveRDS(rf_model, file = "models/saved_models/rf_best_model.rds")
} else {
  xgb.save(xgb_model, "models/saved_models/xgb_best_model.model")
}