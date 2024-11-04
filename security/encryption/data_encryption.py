import os
import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

# Constants for encryption
BLOCK_SIZE = 16
KEY_SIZE = 32  # AES-256 requires a 32-byte key
SALT_SIZE = 16
ITERATIONS = 100000

# Padding for block size
def pad(data):
    padding_length = BLOCK_SIZE - len(data) % BLOCK_SIZE
    padding = chr(padding_length) * padding_length
    return data + padding.encode()

def unpad(data):
    padding_length = data[-1]
    return data[:-padding_length]

# Derive encryption key from password
def derive_key(password, salt, iterations=ITERATIONS, key_size=KEY_SIZE):
    key = PBKDF2(password, salt, dkLen=key_size, count=iterations, hmac_hash_module=hashlib.sha256)
    return key

# Encrypt data using AES
def encrypt_data(plaintext, password):
    salt = get_random_bytes(SALT_SIZE)
    key = derive_key(password, salt)
    
    iv = get_random_bytes(BLOCK_SIZE)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    padded_data = pad(plaintext.encode())
    ciphertext = cipher.encrypt(padded_data)
    
    encrypted_data = base64.b64encode(salt + iv + ciphertext).decode()
    return encrypted_data

# Decrypt data using AES
def decrypt_data(encrypted_data, password):
    encrypted_data_bytes = base64.b64decode(encrypted_data)
    
    salt = encrypted_data_bytes[:SALT_SIZE]
    iv = encrypted_data_bytes[SALT_SIZE:SALT_SIZE + BLOCK_SIZE]
    ciphertext = encrypted_data_bytes[SALT_SIZE + BLOCK_SIZE:]
    
    key = derive_key(password, salt)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    padded_plaintext = cipher.decrypt(ciphertext)
    plaintext = unpad(padded_plaintext).decode()
    
    return plaintext

# Encrypting files
def encrypt_file(input_file_path, output_file_path, password):
    with open(input_file_path, 'rb') as file:
        file_data = file.read()
    
    encrypted_data = encrypt_data(file_data.decode(), password)
    
    with open(output_file_path, 'w') as file:
        file.write(encrypted_data)

# Decrypting files
def decrypt_file(input_file_path, output_file_path, password):
    with open(input_file_path, 'r') as file:
        encrypted_data = file.read()
    
    decrypted_data = decrypt_data(encrypted_data, password)
    
    with open(output_file_path, 'w') as file:
        file.write(decrypted_data)

# File encryption with error handling
def secure_encrypt_file(input_file_path, output_file_path, password):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file {input_file_path} not found.")
    if not password:
        raise ValueError("Password cannot be empty.")
    
    try:
        encrypt_file(input_file_path, output_file_path, password)
        print(f"File encrypted successfully: {output_file_path}")
    except Exception as e:
        print(f"Encryption failed: {e}")

# File decryption with error handling
def secure_decrypt_file(input_file_path, output_file_path, password):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file {input_file_path} not found.")
    if not password:
        raise ValueError("Password cannot be empty.")
    
    try:
        decrypt_file(input_file_path, output_file_path, password)
        print(f"File decrypted successfully: {output_file_path}")
    except Exception as e:
        print(f"Decryption failed: {e}")

# Hashing passwords securely for storage
def hash_password(password):
    salt = get_random_bytes(SALT_SIZE)
    hashed_password = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, ITERATIONS)
    return base64.b64encode(salt + hashed_password).decode()

# Verifying a hashed password
def verify_password(stored_hash, password):
    decoded_hash = base64.b64decode(stored_hash)
    salt = decoded_hash[:SALT_SIZE]
    stored_password_hash = decoded_hash[SALT_SIZE:]
    
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, ITERATIONS)
    
    return stored_password_hash == password_hash

# Symmetric key encryption for API requests
def api_encrypt_request(data, api_key):
    key = hashlib.sha256(api_key.encode()).digest()  # Using SHA-256 to derive key from API key
    iv = get_random_bytes(BLOCK_SIZE)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    padded_data = pad(data.encode())
    ciphertext = cipher.encrypt(padded_data)
    
    return base64.b64encode(iv + ciphertext).decode()

# Symmetric key decryption for API responses
def api_decrypt_response(encrypted_data, api_key):
    key = hashlib.sha256(api_key.encode()).digest()
    encrypted_data_bytes = base64.b64decode(encrypted_data)
    
    iv = encrypted_data_bytes[:BLOCK_SIZE]
    ciphertext = encrypted_data_bytes[BLOCK_SIZE:]
    
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_data = cipher.decrypt(ciphertext)
    
    return unpad(padded_data).decode()

# Securely wiping data from memory
def wipe_memory(data):
    """ Overwrite data in memory with zeros before releasing it. """
    length = len(data)
    zero_data = bytearray(length)
    
    for i in range(length):
        data[i] = zero_data[i]

# API route for encryption and decryption
def handle_api_request(request, api_key):
    if 'data' not in request or 'operation' not in request:
        raise ValueError("Invalid API request format.")
    
    data = request['data']
    operation = request['operation']
    
    if operation == 'encrypt':
        return api_encrypt_request(data, api_key)
    elif operation == 'decrypt':
        return api_decrypt_response(data, api_key)
    else:
        raise ValueError("Unsupported operation. Only 'encrypt' and 'decrypt' are allowed.")

# Testing the encryption functions
def run_tests():
    password = "strong_password"
    data = "Sensitive information to encrypt"
    
    encrypted_data = encrypt_data(data, password)
    decrypted_data = decrypt_data(encrypted_data, password)
    
    assert data == decrypted_data, "Encryption/Decryption failed!"
    
    print("Encryption and decryption tests passed.")

if __name__ == "__main__":
    run_tests()
    input_file = 'website.com/input.txt'
    encrypted_file = 'website.com/encrypted.txt'
    decrypted_file = 'website.com/decrypted.txt'
    password = 'password123'
    
    secure_encrypt_file(input_file, encrypted_file, password)
    secure_decrypt_file(encrypted_file, decrypted_file, password)