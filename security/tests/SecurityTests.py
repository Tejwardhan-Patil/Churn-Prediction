import unittest
import jwt
import os
import tempfile
from security.authentication.jwt_auth import authenticate_token
from security.access_control.rbac import RoleBasedAccessControl
from security.encryption.data_encryption import encrypt_data, decrypt_data
from security.audit.audit_log import AuditLog


class TestAuthentication(unittest.TestCase):

    def setUp(self):
        # Test setup: Load secret key and sample JWT payload
        self.secret_key = 'test_secret_key'
        self.sample_payload = {
            'user_id': 123,
            'role': 'admin',
            'exp': 9999999999  # A far-future expiration
        }
        self.sample_token = jwt.encode(self.sample_payload, self.secret_key, algorithm='HS256')

    def test_valid_token_authentication(self):
        # Test a valid JWT token authentication
        result = authenticate_token(self.sample_token, self.secret_key)
        self.assertTrue(result['authenticated'])
        self.assertEqual(result['user_id'], 123)
        self.assertEqual(result['role'], 'admin')

    def test_invalid_token_authentication(self):
        # Test an invalid JWT token (wrong secret key)
        invalid_secret = 'wrong_secret'
        with self.assertRaises(jwt.exceptions.InvalidSignatureError):
            authenticate_token(self.sample_token, invalid_secret)

    def test_expired_token(self):
        # Test expired token handling
        expired_payload = self.sample_payload.copy()
        expired_payload['exp'] = 0  # Set an expired timestamp
        expired_token = jwt.encode(expired_payload, self.secret_key, algorithm='HS256')
        result = authenticate_token(expired_token, self.secret_key)
        self.assertFalse(result['authenticated'])


class TestRoleBasedAccessControl(unittest.TestCase):

    def setUp(self):
        # Test setup: Define roles and permissions
        self.rbac = RoleBasedAccessControl()
        self.rbac.add_role('admin', permissions=['read', 'write', 'delete'])
        self.rbac.add_role('user', permissions=['read'])

    def test_admin_permissions(self):
        # Test an admin role's permissions
        self.assertTrue(self.rbac.has_permission('admin', 'write'))
        self.assertTrue(self.rbac.has_permission('admin', 'delete'))

    def test_user_permissions(self):
        # Test a normal user's permissions
        self.assertTrue(self.rbac.has_permission('user', 'read'))
        self.assertFalse(self.rbac.has_permission('user', 'delete'))

    def test_invalid_role(self):
        # Test non-existent role handling
        with self.assertRaises(ValueError):
            self.rbac.has_permission('guest', 'read')


class TestEncryption(unittest.TestCase):

    def setUp(self):
        # Test setup: Define a test encryption key and plaintext
        self.encryption_key = 'test_encryption_key'
        self.plaintext = "Sensitive customer data"
        self.encrypted_data = encrypt_data(self.plaintext, self.encryption_key)

    def test_encryption(self):
        # Test that data is encrypted properly
        self.assertNotEqual(self.plaintext, self.encrypted_data)
        self.assertTrue(len(self.encrypted_data) > 0)

    def test_decryption(self):
        # Test that encrypted data can be decrypted
        decrypted_data = decrypt_data(self.encrypted_data, self.encryption_key)
        self.assertEqual(self.plaintext, decrypted_data)

    def test_invalid_decryption(self):
        # Test that decryption with the wrong key fails
        invalid_key = 'wrong_key'
        with self.assertRaises(ValueError):
            decrypt_data(self.encrypted_data, invalid_key)


class TestAuditLog(unittest.TestCase):

    def setUp(self):
        # Test setup: Create a temporary log file
        self.log_file = tempfile.NamedTemporaryFile(delete=False)
        self.audit_log = AuditLog(self.log_file.name)

    def tearDown(self):
        # Cleanup: Remove the temporary log file after each test
        os.unlink(self.log_file.name)

    def test_log_entry(self):
        # Test that an entry is logged correctly
        log_message = "User 123 performed action X"
        self.audit_log.log(log_message)

        # Read log file and verify the entry is recorded
        with open(self.log_file.name, 'r') as f:
            log_content = f.read()
        self.assertIn(log_message, log_content)

    def test_multiple_log_entries(self):
        # Test logging of multiple entries
        messages = ["Entry 1", "Entry 2", "Entry 3"]
        for msg in messages:
            self.audit_log.log(msg)

        # Verify all entries are present in the log file
        with open(self.log_file.name, 'r') as f:
            log_content = f.read()
        for msg in messages:
            self.assertIn(msg, log_content)

    def test_empty_log_entry(self):
        # Test logging of an empty message
        with self.assertRaises(ValueError):
            self.audit_log.log("")


if __name__ == '__main__':
    unittest.main()