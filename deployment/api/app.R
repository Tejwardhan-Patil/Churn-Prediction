library(plumber)
library(jsonlite)
library(data.table)
library(randomForest)
library(glmnet)

# Load the trained model
load("models/saved_model.RData")

#* @apiTitle Churn Prediction API
#* @apiDescription This API predicts customer churn based on input features.

#* Health Check Endpoint
#* @get /health
#* @serializer unboxedJSON
#* @response 200 Returns OK if the API is up and running
function() {
  list(status = "API is up and running", timestamp = Sys.time())
}

# Function to preprocess the input data
preprocess_data <- function(data) {
  # Check for missing data and replace with median values
  for (col in names(data)) {
    if (any(is.na(data[[col]]))) {
      data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
    }
  }
  
  # Scale and normalize features
  scaling_params <- read.csv("data/scaling_params.csv")
  for (col in names(scaling_params)) {
    data[[col]] <- (data[[col]] - scaling_params[col, "min"]) / 
                   (scaling_params[col, "max"] - scaling_params[col, "min"])
  }
  
  return(data)
}

#* Predict Churn
#* @post /predict
#* @param input_data:json The JSON payload containing customer features
#* @response 200 A JSON object containing the churn prediction
#* @serializer unboxedJSON
function(input_data) {
  # Parse input data from JSON
  input_data <- fromJSON(input_data)
  
  # Convert input data to data.table for faster processing
  input_data_dt <- as.data.table(input_data)
  
  # Preprocess the input data
  preprocessed_data <- preprocess_data(input_data_dt)
  
  # Use the loaded model to make predictions
  prediction <- predict(loaded_model, newdata = preprocessed_data, type = "prob")
  
  # Prepare the response
  result <- list(
    customer_id = input_data$customer_id,
    churn_probability = round(prediction[,2], 4) 
  )
  
  return(result)
}

#* Batch Prediction Endpoint
#* @post /batch_predict
#* @param input_data:json A JSON payload containing a list of customers
#* @response 200 A JSON object containing batch predictions
#* @serializer unboxedJSON
function(input_data) {
  # Parse input data from JSON
  input_data <- fromJSON(input_data)
  
  # Convert input data to data.table
  input_data_dt <- as.data.table(input_data)
  
  # Preprocess the data
  preprocessed_data <- preprocess_data(input_data_dt)
  
  # Make batch predictions
  predictions <- predict(loaded_model, newdata = preprocessed_data, type = "prob")
  
  # Prepare batch results
  results <- list()
  for (i in seq_len(nrow(preprocessed_data))) {  # Use seq_len for safety
    result <- list(
      customer_id = input_data$customer_id[i],
      churn_probability = round(predictions[i,2], 4)  
    )
    results[[i]] <- result
  }
  
  return(results)
}

#* Feature Importance Endpoint
#* @get /feature_importance
#* @serializer unboxedJSON
#* @response 200 Returns feature importance from the trained model
function() {
  if (inherits(loaded_model, "randomForest")) {
    importance <- importance(loaded_model)
    importance_df <- data.table(Feature = rownames(importance), Importance = importance[, "MeanDecreaseGini"])
    return(importance_df)
  } else if (inherits(loaded_model, "cv.glmnet")) {
    coef_mat <- coef(loaded_model, s = "lambda.min")
    importance_df <- data.table(Feature = rownames(coef_mat), Importance = abs(coef_mat[, 1]))
    return(importance_df)
  } else {
    return(list(error = "Model type not supported for feature importance."))
  }
}

#* Log request information
#* @filter log
function(req) {
  cat(as.character(Sys.time()), " - ", req$REQUEST_METHOD, " ", req$PATH_INFO, "\n")
  forward()
}

#* Error handling
#* @filter errorHandler
function(req, res) {
  tryCatch({
    forward()
  }, error = function(e) {
    res$status <- 500
    res$body <- list(error = "Internal server error", details = e$message)
    return(res)
  })
}

# Initialize the plumber API
r <- plumb()

# Start the API server
r$run(host = "0.0.0.0", port = 8000)