import time
from locust import HttpUser, task, between, events

# Setting up constants for load test
NUMBER_OF_USERS = 1000  
SPAWN_RATE = 10  
RUN_TIME = "10m"  

class ChurnPredictionLoadTest(HttpUser):
    wait_time = between(1, 5)  

    @task
    def predict_churn(self):
        # Customer data to send in the API request
        customer_data = {
            "customerID": "12345",
            "tenure": 12,
            "monthly_charges": 70.50,
            "total_charges": 840.00,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
            "gender": "Male",
            "partner": "Yes",
            "dependents": "No",
            "phone_service": "Yes",
            "multiple_lines": "No",
            "online_security": "No",
            "online_backup": "Yes",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
        }
        
        # Sending POST request to the churn prediction API endpoint
        with self.client.post("/predict", json=customer_data, catch_response=True) as response:
            # Checking if the API response is successful
            if response.status_code == 200 and "churn" in response.json():
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    # Logging request statistics
    @events.request_success.add_listener
    def log_success(request_type, name, response_time, response_length):
        print(f"SUCCESS: {name}, Response Time: {response_time} ms, Response Length: {response_length} bytes")

    @events.request_failure.add_listener
    def log_failure(request_type, name, response_time, exception):
        print(f"FAILURE: {name}, Response Time: {response_time} ms, Exception: {exception}")

# Running the load test as a script
if __name__ == "__main__":
    import os
    from locust import run_single_user

    # Setting environment variables for Locust load test
    os.environ['LOCUST_USERS'] = str(NUMBER_OF_USERS)
    os.environ['LOCUST_SPAWN_RATE'] = str(SPAWN_RATE)
    os.environ['LOCUST_RUN_TIME'] = RUN_TIME

    # Initiating the test with specified users and spawn rate
    run_single_user(ChurnPredictionLoadTest)