import hashlib
import random
import string
from cryptography.fernet import Fernet
import re

# Load Encryption Key for Fernet (symmetric encryption)
with open("encryption_key.key", "rb") as key_file:
    encryption_key = key_file.read()
cipher_suite = Fernet(encryption_key)

# Utility Functions

def hash_data(input_string: str) -> str:
    """Hashes input data with SHA-256."""
    hashed_value = hashlib.sha256(input_string.encode()).hexdigest()
    return hashed_value

def pseudonymize_data(input_string: str) -> str:
    """Generates a random pseudonym for input data."""
    random.seed(hash_data(input_string))
    pseudonym = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    return pseudonym

def encrypt_data(input_string: str) -> str:
    """Encrypts input data using Fernet symmetric encryption."""
    encrypted_text = cipher_suite.encrypt(input_string.encode())
    return encrypted_text.decode()

def decrypt_data(encrypted_string: str) -> str:
    """Decrypts encrypted data using Fernet symmetric encryption."""
    decrypted_text = cipher_suite.decrypt(encrypted_string.encode())
    return decrypted_text.decode()

# Redaction functions for masking sensitive information

def redact_email(email: str) -> str:
    """Redacts part of the email for privacy."""
    local_part, domain = email.split('@')
    redacted_local = local_part[:2] + "***" + local_part[-1]
    redacted_email = f"{redacted_local}@{domain}"
    return redacted_email

def redact_phone(phone: str) -> str:
    """Redacts all but the last 4 digits of a phone number."""
    redacted_phone = re.sub(r"\d(?=\d{4})", "*", phone)
    return redacted_phone

# Anonymization Function

def anonymize_customer_data(customer_record: dict) -> dict:
    """
    Anonymizes sensitive customer data by applying a series of techniques:
    Pseudonymization, Encryption, and Redaction.
    """
    anonymized_data = {}

    # Pseudonymize customer ID
    if "customer_id" in customer_record:
        anonymized_data["customer_id"] = pseudonymize_data(customer_record["customer_id"])

    # Redact email
    if "email" in customer_record:
        anonymized_data["email"] = redact_email(customer_record["email"])

    # Redact phone number
    if "phone" in customer_record:
        anonymized_data["phone"] = redact_phone(customer_record["phone"])

    # Encrypt sensitive fields
    if "address" in customer_record:
        anonymized_data["address"] = encrypt_data(customer_record["address"])

    if "ssn" in customer_record:
        anonymized_data["ssn"] = encrypt_data(customer_record["ssn"])

    # Hash non-sensitive fields like name
    if "name" in customer_record:
        anonymized_data["name"] = pseudonymize_data(customer_record["name"])

    return anonymized_data

# De-anonymization (Decryption)

def deanonymize_customer_data(anonymized_record: dict) -> dict:
    """
    Reverts the encrypted fields in the anonymized record back to their original values.
    Does not reverse redactions or pseudonymizations.
    """
    deanonymized_data = anonymized_record.copy()

    if "address" in anonymized_record:
        deanonymized_data["address"] = decrypt_data(anonymized_record["address"])

    if "ssn" in anonymized_record:
        deanonymized_data["ssn"] = decrypt_data(anonymized_record["ssn"])

    return deanonymized_data

# Data Validation Function

def validate_customer_data(customer_record: dict) -> bool:
    """
    Validates the integrity of customer data before anonymization.
    Ensures that required fields are present and in correct format.
    """
    required_fields = ["customer_id", "email", "phone", "name", "address", "ssn"]
    for field in required_fields:
        if field not in customer_record:
            return False
    return True

# Usage 

if __name__ == "__main__":
    # Customer Data
    customer_data = {
        "customer_id": "123456",
        "name": "Person1 Doe",
        "email": "person1@website.com",
        "phone": "123-456-7890",
        "address": "1234 Main St, Hometown, HT",
        "ssn": "987-65-4321"
    }

    if validate_customer_data(customer_data):
        # Anonymize the customer data
        anonymized_data = anonymize_customer_data(customer_data)
        print(f"Anonymized Data: {anonymized_data}")

        # De-anonymize to get back sensitive data
        deanonymized_data = deanonymize_customer_data(anonymized_data)
        print(f"De-anonymized Data: {deanonymized_data}")
    else:
        print("Invalid customer data. Cannot proceed with anonymization.")