library(glmnet)
library(randomForest)
library(caret)
library(pROC)
library(testthat)

# Ensure visibility of the global functions
globalVariables(c("roc", "auc", "confusionMatrix"))

# Load dataset
customer_data <- read.csv('../../data/processed/cleaned_customer_data.csv')

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(customer_data$Churn, p = 0.7, list = FALSE)
train_data <- customer_data[trainIndex, ]
test_data <- customer_data[-trainIndex, ]

# Define helper functions for model evaluation
calculate_auc <- function(probabilities, labels) {
  roc_obj <- pROC::roc(labels, probabilities)
  pROC::auc(roc_obj)
}

evaluate_model <- function(model, test_data, model_name) {
  predictions <- predict(model, newdata = test_data, type = "response")
  if (is.factor(predictions)) {
    predictions <- as.numeric(levels(predictions))[predictions]
  }
  
  auc_score <- calculate_auc(predictions, test_data$Churn)
  confusion <- caret::confusionMatrix(as.factor(ifelse(predictions > 0.5, 1, 0)), as.factor(test_data$Churn))
  
  cat("Model:", model_name, "\n")
  cat("AUC Score:", auc_score, "\n")
  cat("Confusion Matrix:\n")
  print(confusion$table)
  
  return(list(auc = auc_score, confusion = confusion))
}

# Load the R models file for testing
source('../../models/r_models.R')

# Test Logistic Regression Model (GLMNET)
test_that("Logistic Regression model (GLMNET) works as expected", {
  glmnet_model <- train_glmnet(train_data)
  results <- evaluate_model(glmnet_model, test_data, "GLMNET Logistic Regression")
  
  expect_true(results$auc > 0.7)
  expect_true(sum(diag(results$confusion$table)) > 200)
})

# Test Random Forest Model
test_that("Random Forest model works as expected", {
  rf_model <- train_random_forest(train_data)
  results <- evaluate_model(rf_model, test_data, "Random Forest")
  
  expect_true(results$auc > 0.8)
  expect_true(sum(diag(results$confusion$table)) > 200)
})

# Test Elastic Net (GLMNET)
test_that("Elastic Net model works as expected", {
  elastic_net_model <- train_elastic_net(train_data)
  results <- evaluate_model(elastic_net_model, test_data, "Elastic Net")
  
  expect_true(results$auc > 0.75)
  expect_true(sum(diag(results$confusion$table)) > 200)
})

# Test model loading from saved RData
test_that("Saved model loading works", {
  load("../../models/saved_models/best_glmnet_model.RData")
  
  results <- evaluate_model(saved_glmnet_model, test_data, "Saved GLMNET Logistic Regression")
  
  expect_true(results$auc > 0.7)
  expect_true(sum(diag(results$confusion$table)) > 200)
})

# Test cross-validation for Random Forest
test_that("Random Forest cross-validation works", {
  rf_cv <- train_random_forest_cross_val(train_data)
  results <- evaluate_model(rf_cv, test_data, "Random Forest Cross-Validation")
  
  expect_true(results$auc > 0.8)
  expect_true(sum(diag(results$confusion$table)) > 200)
})

# Test hyperparameter tuning for GLMNET
test_that("GLMNET hyperparameter tuning works", {
  glmnet_tuned_model <- tune_glmnet(train_data)
  results <- evaluate_model(glmnet_tuned_model, test_data, "GLMNET with Hyperparameter Tuning")
  
  expect_true(results$auc > 0.75)
  expect_true(sum(diag(results$confusion$table)) > 200)
})

# Test hyperparameter tuning for Random Forest
test_that("Random Forest hyperparameter tuning works", {
  rf_tuned_model <- tune_random_forest(train_data)
  results <- evaluate_model(rf_tuned_model, test_data, "Random Forest with Hyperparameter Tuning")
  
  expect_true(results$auc > 0.85)
  expect_true(sum(diag(results$confusion$table)) > 200)
})

# Test feature importance for Random Forest
test_that("Feature importance extraction from Random Forest works", {
  rf_model <- train_random_forest(train_data)
  importance <- varImp(rf_model)
  
  cat("Random Forest Feature Importance:\n")
  print(importance)
  
  expect_true(is.data.frame(importance))
  expect_true(nrow(importance) > 1)
})

# Test GLMNET coefficients
test_that("GLMNET coefficients extraction works", {
  glmnet_model <- train_glmnet(train_data)
  coefficients <- coef(glmnet_model, s = "lambda.min")
  
  cat("GLMNET Coefficients:\n")
  print(coefficients)
  
  expect_true(is.matrix(coefficients))
  expect_true(ncol(coefficients) > 1)
})

# Test ROC curve plotting for Random Forest
test_that("ROC curve for Random Forest works", {
  rf_model <- train_random_forest(train_data)
  predictions <- predict(rf_model, newdata = test_data, type = "response")
  
  roc_obj <- pROC::roc(test_data$Churn, predictions)
  plot(roc_obj, main = "ROC Curve for Random Forest")
  
  auc_score <- pROC::auc(roc_obj)
  cat("AUC Score for Random Forest:", auc_score, "\n")
  
  expect_true(auc_score > 0.8)
})

# Test ROC curve plotting for GLMNET
test_that("ROC curve for GLMNET works", {
  glmnet_model <- train_glmnet(train_data)
  predictions <- predict(glmnet_model, newdata = test_data, type = "response")
  
  roc_obj <- pROC::roc(test_data$Churn, predictions)
  plot(roc_obj, main = "ROC Curve for GLMNET Logistic Regression")
  
  auc_score <- pROC::auc(roc_obj)
  cat("AUC Score for GLMNET:", auc_score, "\n")
  
  expect_true(auc_score > 0.75)
})

# Test if model throws an error on invalid data
test_that("Random Forest throws error on invalid data", {
  expect_error(train_random_forest(NULL))
})

# Test if GLMNET throws error on invalid data
test_that("GLMNET throws error on invalid data", {
  expect_error(train_glmnet(NULL))
})

# Cleanup environment after tests
rm(list = ls())