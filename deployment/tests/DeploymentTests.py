import unittest
import requests
import json
from requests.exceptions import RequestException
import time

class TestDeployment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://localhost:5000" 
        cls.headers = {'Content-Type': 'application/json'}
        cls.health_endpoint = "/health"
        cls.predict_endpoint = "/predict"
        cls.test_payload = {
            "customer_id": 12345,
            "features": {
                "age": 25,
                "monthly_spend": 75.50,
                "contract_type": "month-to-month",
                "tenure": 12,
                "payment_method": "credit_card"
            }
        }

    def test_health_check(self):
        """Test if the API health check returns a 200 status code."""
        try:
            response = requests.get(f"{self.base_url}{self.health_endpoint}")
            self.assertEqual(response.status_code, 200, "Health check failed")
            self.assertEqual(response.json().get("status"), "healthy", "API is not healthy")
        except RequestException as e:
            self.fail(f"Health check failed with exception: {str(e)}")

    def test_predict_endpoint(self):
        """Test the predict endpoint with valid input data."""
        try:
            response = requests.post(
                f"{self.base_url}{self.predict_endpoint}",
                data=json.dumps(self.test_payload),
                headers=self.headers
            )
            self.assertEqual(response.status_code, 200, "Prediction failed")
            self.assertIn("prediction", response.json(), "No prediction returned")
            prediction = response.json().get("prediction")
            self.assertIn(prediction, [0, 1], "Invalid prediction value")
        except RequestException as e:
            self.fail(f"Prediction request failed with exception: {str(e)}")

    def test_predict_endpoint_invalid_input(self):
        """Test the predict endpoint with invalid input data."""
        invalid_payload = {
            "customer_id": None,  # Invalid customer_id
            "features": {
                "age": -1,  # Invalid age
                "monthly_spend": "invalid",  # Invalid monthly spend
                "contract_type": "unknown",  # Invalid contract type
                "tenure": "NaN",  # Invalid tenure
                "payment_method": 123  # Invalid payment method
            }
        }
        try:
            response = requests.post(
                f"{self.base_url}{self.predict_endpoint}",
                data=json.dumps(invalid_payload),
                headers=self.headers
            )
            self.assertEqual(response.status_code, 400, "Invalid input should return 400")
            self.assertIn("error", response.json(), "Error message not returned")
        except RequestException as e:
            self.fail(f"Invalid input test failed with exception: {str(e)}")

    def test_response_time(self):
        """Test if the API responds within the acceptable time frame."""
        max_response_time = 3  # seconds
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}{self.predict_endpoint}",
                data=json.dumps(self.test_payload),
                headers=self.headers
            )
            end_time = time.time()
            response_time = end_time - start_time
            self.assertLessEqual(response_time, max_response_time, f"Response time exceeded {max_response_time} seconds")
        except RequestException as e:
            self.fail(f"Response time test failed with exception: {str(e)}")

    def test_multiple_predictions(self):
        """Test if the API can handle multiple prediction requests in a short time frame."""
        number_of_requests = 10
        for i in range(number_of_requests):
            try:
                response = requests.post(
                    f"{self.base_url}{self.predict_endpoint}",
                    data=json.dumps(self.test_payload),
                    headers=self.headers
                )
                self.assertEqual(response.status_code, 200, f"Failed on request {i+1}")
                self.assertIn("prediction", response.json(), "No prediction returned")
            except RequestException as e:
                self.fail(f"Multiple predictions test failed with exception: {str(e)}")

    def test_model_version_endpoint(self):
        """Test if the model version endpoint returns correct version information."""
        model_version_endpoint = "/model/version"
        try:
            response = requests.get(f"{self.base_url}{model_version_endpoint}")
            self.assertEqual(response.status_code, 200, "Model version request failed")
            self.assertIn("version", response.json(), "Model version not found")
            model_version = response.json().get("version")
            self.assertTrue(isinstance(model_version, str), "Invalid version format")
        except RequestException as e:
            self.fail(f"Model version endpoint test failed with exception: {str(e)}")

    def test_predict_large_payload(self):
        """Test the predict endpoint with a large payload."""
        large_payload = {
            "customer_id": 54321,
            "features": {
                "age": 30,
                "monthly_spend": 100.00,
                "contract_type": "yearly",
                "tenure": 36,
                "payment_method": "debit_card",
                "additional_info": "x" * 1000000  # Simulating large payload with long string
            }
        }
        try:
            response = requests.post(
                f"{self.base_url}{self.predict_endpoint}",
                data=json.dumps(large_payload),
                headers=self.headers
            )
            self.assertEqual(response.status_code, 200, "Large payload prediction failed")
            self.assertIn("prediction", response.json(), "No prediction returned for large payload")
        except RequestException as e:
            self.fail(f"Large payload test failed with exception: {str(e)}")

    def test_authentication_required(self):
        """Test that API endpoints require authentication if implemented."""
        auth_headers = self.headers.copy()
        auth_headers["Authorization"] = "Bearer invalid_token"
        try:
            response = requests.post(
                f"{self.base_url}{self.predict_endpoint}",
                data=json.dumps(self.test_payload),
                headers=auth_headers
            )
            self.assertEqual(response.status_code, 401, "Unauthenticated request should return 401")
        except RequestException as e:
            self.fail(f"Authentication test failed with exception: {str(e)}")

    def test_api_logging(self):
        """Test if the API logs the requests correctly."""
        log_endpoint = "/logs/recent"
        try:
            response = requests.get(f"{self.base_url}{log_endpoint}")
            self.assertEqual(response.status_code, 200, "Logs endpoint request failed")
            logs = response.json().get("logs")
            self.assertTrue(isinstance(logs, list), "Logs not returned correctly")
        except RequestException as e:
            self.fail(f"API logging test failed with exception: {str(e)}")

    def test_prediction_consistency(self):
        """Test if the model provides consistent results for the same input."""
        try:
            first_response = requests.post(
                f"{self.base_url}{self.predict_endpoint}",
                data=json.dumps(self.test_payload),
                headers=self.headers
            )
            second_response = requests.post(
                f"{self.base_url}{self.predict_endpoint}",
                data=json.dumps(self.test_payload),
                headers=self.headers
            )
            self.assertEqual(first_response.status_code, 200, "First prediction failed")
            self.assertEqual(second_response.status_code, 200, "Second prediction failed")
            first_prediction = first_response.json().get("prediction")
            second_prediction = second_response.json().get("prediction")
            self.assertEqual(first_prediction, second_prediction, "Predictions are inconsistent")
        except RequestException as e:
            self.fail(f"Prediction consistency test failed with exception: {str(e)}")


if __name__ == '__main__':
    unittest.main()