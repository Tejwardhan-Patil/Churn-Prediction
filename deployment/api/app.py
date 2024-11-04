from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np
import logging
from logging.config import dictConfig
import os

# Initialize Flask application
app = Flask(__name__)

# Configure logging
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models', 'best_model.pkl')
model = joblib.load(model_path)

# Load the scaler
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'features', 'scaling_normalization.pkl')
scaler = joblib.load(scaler_path)

# API route to check if the service is running
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is up and running"}), 200

# Helper function to preprocess input data
def preprocess_data(data):
    try:
        # Convert input data to pandas DataFrame
        input_data = pd.DataFrame([data])

        # Apply feature scaling using the preloaded scaler
        scaled_data = scaler.transform(input_data)
        return scaled_data
    except Exception as e:
        app.logger.error(f"Error preprocessing data: {str(e)}")
        return None

# API route for churn prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from JSON request
        input_json = request.json
        app.logger.info(f"Received input: {input_json}")

        # Extract relevant fields for prediction
        customer_data = {
            'customer_id': input_json.get('customer_id'),
            'tenure': input_json.get('tenure'),
            'monthly_charges': input_json.get('monthly_charges'),
            'total_charges': input_json.get('total_charges'),
            'contract_type': input_json.get('contract_type'),
            'payment_method': input_json.get('payment_method')
        }

        # Preprocess the input data
        preprocessed_data = preprocess_data(customer_data)
        if preprocessed_data is None:
            return jsonify({'error': 'Preprocessing failed'}), 500

        # Generate prediction using the loaded model
        prediction = model.predict(preprocessed_data)
        prediction_prob = model.predict_proba(preprocessed_data)

        # Create a response
        response = {
            'customer_id': customer_data['customer_id'],
            'prediction': int(prediction[0]),
            'probability': prediction_prob[0].tolist()
        }

        return jsonify(response), 200
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

# API route to get model information
@app.route('/model_info', methods=['GET'])
def model_info():
    try:
        model_details = {
            'model_type': type(model).__name__,
            'model_version': '1.0',
            'features_used': ['tenure', 'monthly_charges', 'total_charges', 'contract_type', 'payment_method']
        }
        return jsonify(model_details), 200
    except Exception as e:
        app.logger.error(f"Error fetching model info: {str(e)}")
        return jsonify({'error': 'Unable to fetch model info'}), 500

# API route for bulk predictions
@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    try:
        # Parse input data from JSON request
        input_json = request.json
        app.logger.info(f"Received bulk input: {input_json}")

        # Convert input data to pandas DataFrame
        input_df = pd.DataFrame(input_json['data'])

        # Preprocess the input data
        preprocessed_data = scaler.transform(input_df)

        # Generate predictions for bulk data
        predictions = model.predict(preprocessed_data)
        prediction_probs = model.predict_proba(preprocessed_data)

        # Create a response for bulk predictions
        response = []
        for i, customer_id in enumerate(input_df['customer_id']):
            response.append({
                'customer_id': customer_id,
                'prediction': int(predictions[i]),
                'probability': prediction_probs[i].tolist()
            })

        return jsonify(response), 200
    except Exception as e:
        app.logger.error(f"Error in bulk prediction: {str(e)}")
        return jsonify({'error': 'Bulk prediction failed'}), 500

# API route for handling input validation
@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({'error': 'Bad Request - Invalid Input'}), 400

# API route for handling internal server errors
@app.errorhandler(500)
def handle_internal_error(e):
    return jsonify({'error': 'Internal Server Error'}), 500

# Main entry point for running the app
if __name__ == "__main__":
    # Set host and port dynamically via environment variables or default values
    app.run(host=os.getenv('FLASK_RUN_HOST', '0.0.0.0'), port=int(os.getenv('FLASK_RUN_PORT', 5000)))