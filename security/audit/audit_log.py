import os
import logging
from datetime import datetime
import json
import hashlib
from cryptography.fernet import Fernet

# Configure logging
LOG_FILE = "audit_log.json"
logging.basicConfig(filename="audit_errors.log", level=logging.ERROR, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Generate or load encryption key
KEY_FILE = "secret.key"
if not os.path.exists(KEY_FILE):
    key = Fernet.generate_key()
    with open(KEY_FILE, 'wb') as key_file:
        key_file.write(key)
else:
    with open(KEY_FILE, 'rb') as key_file:
        key = key_file.read()

cipher_suite = Fernet(key)


class AuditLog:
    def __init__(self, log_file=LOG_FILE):
        self.log_file = log_file
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                json.dump([], f)

    def _get_current_timestamp(self):
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    def _calculate_hash(self, data):
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _encrypt(self, message):
        return cipher_suite.encrypt(message.encode('utf-8')).decode('utf-8')

    def _decrypt(self, encrypted_message):
        return cipher_suite.decrypt(encrypted_message.encode('utf-8')).decode('utf-8')

    def log_access(self, user_id, action, resource, details=None):
        timestamp = self._get_current_timestamp()
        data = {
            'timestamp': timestamp,
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'details': details or "",
        }
        data['hash'] = self._calculate_hash(json.dumps(data))
        data_encrypted = self._encrypt(json.dumps(data))

        self._write_to_log(data_encrypted)

    def log_modification(self, user_id, action, resource, before_change, after_change):
        timestamp = self._get_current_timestamp()
        data = {
            'timestamp': timestamp,
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'before_change': before_change,
            'after_change': after_change
        }
        data['hash'] = self._calculate_hash(json.dumps(data))
        data_encrypted = self._encrypt(json.dumps(data))

        self._write_to_log(data_encrypted)

    def _write_to_log(self, log_entry):
        try:
            with open(self.log_file, 'r') as log:
                logs = json.load(log)
        except Exception as e:
            logging.error(f"Error reading log file: {str(e)}")
            logs = []

        logs.append(log_entry)
        try:
            with open(self.log_file, 'w') as log:
                json.dump(logs, log)
        except Exception as e:
            logging.error(f"Error writing to log file: {str(e)}")

    def verify_log(self):
        try:
            with open(self.log_file, 'r') as log:
                logs = json.load(log)
        except Exception as e:
            logging.error(f"Error reading log file: {str(e)}")
            return []

        valid_logs = []
        for log_entry in logs:
            decrypted_log = json.loads(self._decrypt(log_entry))
            log_hash = decrypted_log.pop('hash', None)
            if log_hash and log_hash == self._calculate_hash(json.dumps(decrypted_log)):
                valid_logs.append(decrypted_log)
            else:
                logging.error("Audit log tampering detected.")

        return valid_logs

    def get_logs(self, user_id=None, resource=None, action=None):
        try:
            with open(self.log_file, 'r') as log:
                logs = json.load(log)
        except Exception as e:
            logging.error(f"Error reading log file: {str(e)}")
            return []

        filtered_logs = []
        for log_entry in logs:
            decrypted_log = json.loads(self._decrypt(log_entry))

            if user_id and decrypted_log['user_id'] != user_id:
                continue
            if resource and decrypted_log['resource'] != resource:
                continue
            if action and decrypted_log['action'] != action:
                continue

            filtered_logs.append(decrypted_log)

        return filtered_logs


class AuditMiddleware:
    """
    Middleware to log actions to the audit log.
    Can be integrated into an API or web framework to track API access and actions.
    """
    def __init__(self, audit_logger):
        self.audit_logger = audit_logger

    def log_request(self, request):
        user_id = request.headers.get('X-User-ID')
        resource = request.path
        action = request.method
        self.audit_logger.log_access(user_id, action, resource)

    def log_modification(self, request, before_change, after_change):
        user_id = request.headers.get('X-User-ID')
        resource = request.path
        action = "MODIFY"
        self.audit_logger.log_modification(user_id, action, resource, before_change, after_change)


# Usage
if __name__ == "__main__":
    audit_log = AuditLog()

    # logging access
    audit_log.log_access(user_id="1234", action="READ", resource="customer_data", details="Accessed customer data.")

    # logging modification
    audit_log.log_modification(user_id="1234", action="UPDATE", resource="customer_data", 
                               before_change={"name": "Person1"}, after_change={"name": "Person2"})

    # verification
    valid_logs = audit_log.verify_log()
    print("Valid Logs:", valid_logs)

    # Fetch logs filtered by user_id
    user_logs = audit_log.get_logs(user_id="1234")
    print("Logs for user 1234:", user_logs)