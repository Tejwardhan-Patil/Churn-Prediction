library(glmnet)
library(randomForest)
library(xgboost)
library(caret)
library(tidyverse)
library(pROC)
library(ggplot2)

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
glmnet_model <- glmnet(x_train, y_train, family = "binomial", lambda = best_lambda, alpha = 0.5)
glmnet_probs <- predict(glmnet_model, newx = x_test, type = "response")
glmnet_pred <- ifelse(glmnet_probs > 0.5, 1, 0)

# Random Forest Model
rf_model <- randomForest(Churn ~ ., data = train_data, ntree = 500, mtry = 3, importance = TRUE)
rf_pred <- predict(rf_model, newdata = test_data)
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[, 2]

# XGBoost Model
dtrain <- xgb.DMatrix(data = x_train, label = as.numeric(y_train) - 1)
dtest <- xgb.DMatrix(data = x_test, label = as.numeric(y_test) - 1)
xgb_params <- list(objective = "binary:logistic", eval_metric = "auc", eta = 0.1, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8)
xgb_cv <- xgb.cv(params = xgb_params, data = dtrain, nfold = 10, nrounds = 100, early_stopping_rounds = 10, verbose = 0)
best_nrounds <- xgb_cv$best_iteration
xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = best_nrounds)
xgb_probs <- predict(xgb_model, newdata = dtest)
xgb_pred <- ifelse(xgb_probs > 0.5, 1, 0)

# Evaluation Function (Accuracy, AUC, Confusion Matrix, Precision, Recall, F1)
evaluate_model <- function(true_labels, predicted_probs, predicted_classes, model_name) {
  auc_value <- pROC::auc(pROC::roc(true_labels, predicted_probs))
  accuracy <- mean(predicted_classes == true_labels)
  confusion <- caret::confusionMatrix(as.factor(predicted_classes), true_labels)
  precision <- caret::posPredValue(as.factor(predicted_classes), true_labels)
  recall <- caret::sensitivity(as.factor(predicted_classes), true_labels)
  f1_score <- (2 * precision * recall) / (precision + recall)
  
  cat(paste0("\n", model_name, " Performance:\n"))
  cat("Accuracy:", accuracy, "\n")
  cat("AUC:", auc_value, "\n")
  cat("Precision:", precision, "\n")
  cat("Recall:", recall, "\n")
  cat("F1 Score:", f1_score, "\n")
  cat("Confusion Matrix:\n")
  print(confusion)
  
  return(list(accuracy = accuracy, auc = auc_value, precision = precision, recall = recall, f1 = f1_score))
}

# Evaluate GLMNET Model
glmnet_metrics <- evaluate_model(y_test, glmnet_probs, glmnet_pred, "GLMNET")

# Evaluate Random Forest Model
rf_metrics <- evaluate_model(y_test, rf_probs, rf_pred, "Random Forest")

# Evaluate XGBoost Model
xgb_metrics <- evaluate_model(y_test, xgb_probs, xgb_pred, "XGBoost")

# Model Comparison and Visualization
performance_df <- data.frame(
  Model = c("GLMNET", "Random Forest", "XGBoost"),
  Accuracy = c(glmnet_metrics$accuracy, rf_metrics$accuracy, xgb_metrics$accuracy),
  AUC = c(glmnet_metrics$auc, rf_metrics$auc, xgb_metrics$auc),
  Precision = c(glmnet_metrics$precision, rf_metrics$precision, xgb_metrics$precision),
  Recall = c(glmnet_metrics$recall, rf_metrics$recall, xgb_metrics$recall),
  F1_Score = c(glmnet_metrics$f1, rf_metrics$f1, xgb_metrics$f1)
)

cat("\nModel Comparison:\n")
print(performance_df)

# Visualizing Model Comparison
ggplot(performance_df, aes(x = Model)) +
  geom_bar(aes(y = AUC, fill = "AUC"), stat = "identity", position = "dodge") +
  geom_bar(aes(y = Accuracy, fill = "Accuracy"), stat = "identity", position = "dodge") +
  geom_bar(aes(y = F1_Score, fill = "F1 Score"), stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison", y = "Metric", fill = "Metric") +
  theme_minimal()

# Save the Best Model Based on AUC
best_model <- performance_df$Model[which.max(performance_df$AUC)]
cat("\nBest Model Selected:", best_model, "\n")

if (best_model == "GLMNET") {
  saveRDS(glmnet_model, file = "models/saved_models/glmnet_best_model.rds")
} else if (best_model == "Random Forest") {
  saveRDS(rf_model, file = "models/saved_models/rf_best_model.rds")
} else {
  xgb.save(xgb_model, "models/saved_models/xgb_best_model.model")
}