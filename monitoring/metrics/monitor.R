library(caret)
library(ggplot2)
library(pROC)
library(randomForest)
library(glmnet)
library(dplyr)
library(rlang)
library(mailR)

# Load Models and Data
load_model <- function(model_path) {
load(model_path)
}

load_data <- function(test_data_path) {
read.csv(test_data_path)
}

# Predictions
make_predictions <- function(model, data) {
predict(model, newdata = data, type = "response")
}

# Metric Calculation Functions
calculate_accuracy <- function(predictions, actual) {
sum(predictions == actual) / length(actual)
}

calculate_precision <- function(predictions, actual) {
pos_pred <- sum(predictions == 1 & actual == 1)
pos_pred / sum(predictions == 1)
}

calculate_recall <- function(predictions, actual) {
true_pos <- sum(predictions == 1 & actual == 1)
true_pos / sum(actual == 1)
}

calculate_f1_score <- function(precision, recall) {
2 * ((precision * recall) / (precision + recall))
}

# Ensure the roc and auc functions are fully qualified
calculate_auc_roc <- function(predictions, actual) {
roc_obj <- pROC::roc(actual, predictions)
pROC::auc(roc_obj)
}

# Log Performance Metrics
log_metrics <- function(metrics, log_file) {
write.table(metrics, file = log_file, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE)
}

# Generate Metrics for Monitoring
generate_metrics <- function(model, test_data, actual) {
predictions <- make_predictions(model, test_data)
accuracy <- calculate_accuracy(predictions, actual)
precision <- calculate_precision(predictions, actual)
recall <- calculate_recall(predictions, actual)
f1 <- calculate_f1_score(precision, recall)
auc_roc <- calculate_auc_roc(predictions, actual)

data.frame(
Accuracy = accuracy,
Precision = precision,
Recall = recall,
F1_Score = f1,
AUC_ROC = auc_roc
)
}

# Performance Drift Detection
check_drift <- function(current_metrics, historical_metrics) {
drift_detected <- FALSE
threshold <- 0.05

for (metric in names(current_metrics)) {
diff <- abs(current_metrics[[metric]] - historical_metrics[[metric]])
if (diff > threshold) {
drift_detected <- TRUE
break
}
}

return(drift_detected)
}

# Alert System for Performance Drops
send_alert <- function(message) {
sender <- "monitoring@website.com"
recipients <- c("admin@website.com")
subject <- "Performance Alert: Model Drift Detected"

# mailR's send.mail function
mailR::send.mail(
from = sender,
to = recipients,
subject = subject,
body = message,
smtp = list(
host.name = "smtp.website.com",
port = 587,
user.name = "monitoring@website.com",
passwd = "password123",
tls = TRUE
),
authenticate = TRUE,
send = TRUE
)
}

# Plotting Metrics Trends
plot_metrics_trend <- function(metrics_log) {
metrics_data <- read.csv(metrics_log)
ggplot(metrics_data, aes(x = .data$Date)) +
geom_line(aes(y = .data$Accuracy, color = "Accuracy")) +
geom_line(aes(y = .data$Precision, color = "Precision")) +
geom_line(aes(y = .data$Recall, color = "Recall")) +
geom_line(aes(y = .data$F1_Score, color = "F1_Score")) +
geom_line(aes(y = .data$AUC_ROC, color = "AUC_ROC")) +
labs(title = "Model Performance Metrics Over Time",
x = "Date", y = "Metric Value") +
theme_minimal()
}

# Run Monitoring
run_monitoring <- function(model_path, test_data_path, actual, log_file, historical_metrics_path) {
model <- load_model(model_path)
test_data <- load_data(test_data_path)

current_metrics <- generate_metrics(model, test_data, actual)
historical_metrics <- read.csv(historical_metrics_path)

# Log Current Metrics
log_metrics(current_metrics, log_file)

# Check for Drift
if (check_drift(current_metrics, historical_metrics)) {
send_alert(paste0("Model performance drift detected! Metrics: \n",
"Accuracy: ", current_metrics$Accuracy, "\n",
"Precision: ", current_metrics$Precision, "\n",
"Recall: ", current_metrics$Recall, "\n",
"F1 Score: ", current_metrics$F1_Score, "\n",
"AUC-ROC: ", current_metrics$AUC_ROC))
} else {
print("No performance drift detected.")
}

# Plot Trend
plot_metrics_trend(log_file)
}

# Monitor Performance on Regular Basis
schedule_monitoring <- function(interval, model_path, test_data_path, actual, log_file, historical_metrics_path) {
while (TRUE) {
run_monitoring(model_path, test_data_path, actual, log_file, historical_metrics_path)
Sys.sleep(interval)
}
}

# Test Monitoring System
test_monitoring <- function() {
model_path <- "models/best_model.RData"
test_data_path <- "data/processed/cleaned_customer_data.csv"
actual <- read.csv("data/processed/actual_labels.csv")$Churn
log_file <- "monitoring/logs/metrics_log.csv"
historical_metrics_path <- "monitoring/logs/historical_metrics.csv"

# Run Monitoring Every Hour
schedule_monitoring(3600, model_path, test_data_path, actual, log_file, historical_metrics_path)
}

# Execute Monitoring
test_monitoring()