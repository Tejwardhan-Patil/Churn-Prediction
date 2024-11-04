# =====================================================
# Package Installation and Loading for R API Deployment
# =====================================================

# Check if the package is installed; if not, install it
install_if_missing <- function(package_name) {
  if (!requireNamespace(package_name, quietly = TRUE)) {
    install.packages(package_name, repos = "https://cloud.r-project.org/")
  }
  library(package_name, character.only = TRUE)
}

# Core Packages
install_if_missing("plumber")     
install_if_missing("jsonlite")     
install_if_missing("httr")        
install_if_missing("readr")       
install_if_missing("dplyr")      
install_if_missing("stringr")     
install_if_missing("tidyr")        
install_if_missing("purrr")        

# Machine Learning and Data Science Packages
install_if_missing("glmnet")       
install_if_missing("randomForest")  
install_if_missing("caret")         
install_if_missing("xgboost")     
install_if_missing("e1071")       
install_if_missing("pROC")        
install_if_missing("Metrics")    

# Model Saving and Serialization
install_if_missing("saveRDS")    
install_if_missing("readRDS")      

# Performance Monitoring and Logging
install_if_missing("logger")       
install_if_missing("prometheus")   
install_if_missing("plumberPrometheus") 
install_if_missing("lubridate")    

# Visualization (for monitoring endpoints)
install_if_missing("ggplot2")      
install_if_missing("plotly")       
install_if_missing("ggcorrplot")  

# API Security
install_if_missing("plumber")     
install_if_missing("jwt")         
install_if_missing("openssl")     
install_if_missing("digest")      

# Database Access and Management
install_if_missing("DBI")        
install_if_missing("RSQLite")    
install_if_missing("odbc")          
install_if_missing("RMariaDB")      
install_if_missing("RPostgreSQL")   

# Docker Integration (for API deployment in containers)
install_if_missing("dockerfiler")   
install_if_missing("packrat")      

# YAML for configuration file management
install_if_missing("yaml")         

# Data Preprocessing (used in models)
install_if_missing("preprocessCore")  
install_if_missing("recipes")        

# Model Explanation and Interpretability
install_if_missing("lime")        
install_if_missing("DALEX")        

# =====================================================
# Load all packages for API deployment and model serving
# =====================================================
library(plumber)
library(jsonlite)
library(httr)
library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(purrr)
library(glmnet)
library(randomForest)
library(caret)
library(xgboost)
library(e1071)
library(pROC)
library(Metrics)
library(logger)
library(prometheus)
library(plumberPrometheus)
library(lubridate)
library(ggplot2)
library(plotly)
library(ggcorrplot)
library(jwt)
library(openssl)
library(digest)
library(DBI)
library(RSQLite)
library(odbc)
library(RMariaDB)
library(RPostgreSQL)
library(dockerfiler)
library(packrat)
library(yaml)
library(preprocessCore)
library(recipes)
library(lime)
library(DALEX)

# =====================================================
# Set up Logging Configuration
# =====================================================
# Initialize logger to record all API activities
log_app <- logger::log_file("logs/api_log.txt")

# Log a message to indicate startup
logger::log_info("R API starting up and loading necessary packages", namespace = "api_startup")

# =====================================================
# Prometheus Metrics Setup
# =====================================================
# Prometheus endpoint for API metrics
pr <- plumber::plumber$new()

# Prometheus integration for performance monitoring
pr$registerHooks(plumberPrometheus::prometheusHooks())

# =====================================================
# Load Pre-trained Models
# =====================================================
model_path <- "models/churn_model.rds"
churn_model <- readRDS(model_path)

# =====================================================
# Start Plumber API
# =====================================================
# Create an API route for prediction
pr$handle("POST", "/predict", function(req, res) {
  # Parse incoming data
  input_data <- jsonlite::fromJSON(req$postBody)
  
  # Preprocess input data 
  processed_data <- as.data.frame(input_data)
  
  # Perform prediction using pre-trained model
  prediction <- predict(churn_model, processed_data, type = "response")
  
  # Return the prediction result as JSON
  return(list(prediction = prediction))
})

# =====================================================
# Start the API and listen on port 8000
# =====================================================
pr$run(host = "0.0.0.0", port = 8000)