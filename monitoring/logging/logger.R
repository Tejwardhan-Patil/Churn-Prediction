library(log4r)

# Create a Logger
logger <- create.logger()
logfile(logger) <- "logs/churn_model.log"
level(logger) <- "INFO"

# Initialize Logging System
log_info <- function(message) {
  log4r::info(logger, message)
}

log_warning <- function(message) {
  log4r::warn(logger, message)
}

log_error <- function(message) {
  log4r::error(logger, message)
}

log_debug <- function(message) {
  log4r::debug(logger, message)
}

log_info("Logger initialized for Churn Prediction Monitoring.")

# Monitoring Function to Check Data Flow
monitor_data_flow <- function(data_path) {
  log_info(paste("Monitoring data flow from", data_path))
  
  if (file.exists(data_path)) {
    log_info(paste("Data found at", data_path))
    
    data_size <- file.info(data_path)$size
    log_info(paste("Data size:", data_size, "bytes"))
    
    if (data_size == 0) {
      log_warning("Data size is 0 bytes. Potential issue with data generation.")
    } else {
      log_info("Data size looks healthy.")
    }
  } else {
    log_error(paste("Data not found at", data_path))
  }
}

# Monitoring Data Pipeline Status
monitor_pipeline_status <- function(stage, status) {
  log_info(paste("Monitoring pipeline status at stage:", stage, "Status:", status))
  
  if (status == "SUCCESS") {
    log_info(paste(stage, "completed successfully."))
  } else if (status == "FAILURE") {
    log_error(paste(stage, "failed. Immediate attention needed!"))
  } else {
    log_warning(paste("Pipeline stage", stage, "has unexpected status:", status))
  }
}

# Monitor Model Performance
monitor_model_performance <- function(model_name, metric_name, value, threshold) {
  log_info(paste("Monitoring model performance for", model_name))
  log_info(paste("Metric:", metric_name, "Value:", value, "Threshold:", threshold))
  
  if (value < threshold) {
    log_warning(paste(model_name, "performance issue detected. Metric:", metric_name, "is below threshold."))
  } else {
    log_info(paste(model_name, "performance is within acceptable range for metric:", metric_name))
  }
}

# Monitoring Data File
monitor_data_flow("data/processed/cleaned_customer_data.csv")

# Monitoring Pipeline Status
monitor_pipeline_status("Feature Engineering", "SUCCESS")
monitor_pipeline_status("Model Training", "FAILURE")

# Monitoring Model Performance
monitor_model_performance("Random Forest", "Accuracy", 0.84, 0.85)

# Monitor API Request Logs
log_api_request <- function(request_method, endpoint, status_code) {
  log_info(paste("API Request - Method:", request_method, "Endpoint:", endpoint, "Status:", status_code))
  
  if (status_code != 200) {
    log_warning(paste("Non-200 response detected:", status_code, "for endpoint", endpoint))
  }
}

# Logging an API Request
log_api_request("POST", "/predict", 500)

# Monitor Data Drift
monitor_data_drift <- function(old_data_path, new_data_path) {
  log_info("Monitoring data drift between two datasets.")
  
  if (!file.exists(old_data_path) || !file.exists(new_data_path)) {
    log_error("One or both datasets not found. Data drift monitoring cannot proceed.")
    return()
  }
  
  old_data <- read.csv(old_data_path)
  new_data <- read.csv(new_data_path)
  
  old_mean <- mean(old_data$churn_rate)
  new_mean <- mean(new_data$churn_rate)
  
  log_info(paste("Old churn rate mean:", old_mean))
  log_info(paste("New churn rate mean:", new_mean))
  
  drift_threshold <- 0.05
  drift_detected <- abs(old_mean - new_mean) > drift_threshold
  
  if (drift_detected) {
    log_warning("Data drift detected. Significant change in churn rate.")
  } else {
    log_info("No significant data drift detected.")
  }
}

# Monitoring Data Drift
monitor_data_drift("data/raw/customer_data.csv", "data/processed/cleaned_customer_data.csv")

# Monitor Model Drift
monitor_model_drift <- function(old_model_path, new_model_path) {
  log_info("Monitoring model drift between two models.")
  
  if (!file.exists(old_model_path) || !file.exists(new_model_path)) {
    log_error("One or both model files not found. Model drift monitoring cannot proceed.")
    return()
  }
  
  old_model <- readRDS(old_model_path)
  new_model <- readRDS(new_model_path)
  
  # RMSE (Root Mean Square Error) to check drift
  old_rmse <- old_model$performance_metrics$rmse
  new_rmse <- new_model$performance_metrics$rmse
  
  log_info(paste("Old model RMSE:", old_rmse))
  log_info(paste("New model RMSE:", new_rmse))
  
  drift_threshold <- 0.1
  drift_detected <- abs(old_rmse - new_rmse) > drift_threshold
  
  if (drift_detected) {
    log_warning("Model drift detected. Performance has degraded.")
  } else {
    log_info("No significant model drift detected.")
  }
}

# Monitoring Model Drift
monitor_model_drift("models/saved_models/old_model.rds", "models/saved_models/new_model.rds")

# Monitor Memory Usage
monitor_memory_usage <- function() {
  mem_usage <- memory.size()
  mem_limit <- memory.limit()
  
  log_info(paste("Memory usage:", mem_usage, "MB of", mem_limit, "MB"))
  
  usage_ratio <- mem_usage / mem_limit
  if (usage_ratio > 0.8) {
    log_warning("Memory usage exceeds 80% of available memory.")
  }
}

# Monitoring Memory Usage
monitor_memory_usage()

# Monitor CPU Usage
monitor_cpu_usage <- function() {
  # Approximation using system command
  cpu_usage <- system("top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1}'", intern=TRUE)
  cpu_usage <- as.numeric(cpu_usage)
  
  log_info(paste("CPU usage:", cpu_usage, "%"))
  
  if (cpu_usage > 85) {
    log_warning("CPU usage exceeds 85%. Potential bottleneck.")
  }
}

# Monitoring CPU Usage
monitor_cpu_usage()