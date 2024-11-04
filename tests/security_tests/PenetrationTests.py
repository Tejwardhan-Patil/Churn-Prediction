import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import paramiko
import time

# Initialize Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)

class PenetrationTests:

    def __init__(self):
        self.base_url = "https://website.com"
        self.ssh_host = "website.com"
        self.ssh_user = "penetration_test_user"
        self.ssh_password = "test_password"

    # Test for SQL Injection Vulnerabilities
    def test_sql_injection(self):
        url = f"{self.base_url}/login"
        payload = {"username": "' OR 1=1 --", "password": "irrelevant"}
        response = requests.post(url, data=payload)
        if "dashboard" in response.url:
            print("[+] SQL Injection vulnerability detected.")
        else:
            print("[-] SQL Injection not detected.")

    # Test for Cross-Site Scripting (XSS) Vulnerabilities
    def test_xss(self):
        url = f"{self.base_url}/search"
        payload = {"query": "<script>alert('XSS')</script>"}
        response = requests.get(url, params=payload)
        if "<script>alert('XSS')</script>" in response.text:
            print("[+] XSS vulnerability detected.")
        else:
            print("[-] XSS not detected.")

    # Test for Open Redirect Vulnerability
    def test_open_redirect(self):
        url = f"{self.base_url}/redirect"
        payload = {"url": "http://malicious-site.com"}
        response = requests.get(url, params=payload)
        if "http://malicious-site.com" in response.url:
            print("[+] Open Redirect vulnerability detected.")
        else:
            print("[-] Open Redirect not detected.")

    # Test for Brute Force Protection in Authentication
    def test_brute_force(self):
        url = f"{self.base_url}/login"
        for i in range(100):  # Simulating brute force
            payload = {"username": "test_user", "password": f"wrong_password{i}"}
            response = requests.post(url, data=payload)
            if "Too many attempts" in response.text:
                print("[+] Brute force protection detected.")
                break
        else:
            print("[-] Brute force protection not detected.")

    # Test for Server Misconfiguration via SSH
    def test_ssh_connection(self):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.ssh_host, username=self.ssh_user, password=self.ssh_password)
            stdin, stdout, stderr = ssh.exec_command("ls /")
            output = stdout.read().decode()
            if "sensitive_data" in output:
                print("[+] Server misconfiguration detected. Sensitive data exposed.")
            else:
                print("[-] No sensitive data exposure via SSH.")
        except Exception as e:
            print(f"[-] SSH connection failed: {e}")

    # Test for Authentication Bypass
    def test_auth_bypass(self):
        driver.get(f"{self.base_url}/admin")
        if "login" not in driver.current_url:
            print("[+] Authentication bypass detected.")
        else:
            print("[-] No authentication bypass detected.")

    # Test for Clickjacking Vulnerabilities
    def test_clickjacking(self):
        headers = {
            "X-Frame-Options": "ALLOWALL"
        }
        response = requests.get(self.base_url, headers=headers)
        if "X-Frame-Options" not in response.headers:
            print("[+] Clickjacking vulnerability detected.")
        else:
            print("[-] Clickjacking not detected.")

    # Test for Insecure HTTP Methods (PUT, DELETE)
    def test_insecure_http_methods(self):
        url = f"{self.base_url}/api/resource"
        methods = ["PUT", "DELETE", "OPTIONS"]
        for method in methods:
            response = requests.request(method, url)
            if response.status_code != 405:
                print(f"[+] Insecure HTTP Method '{method}' allowed.")
            else:
                print(f"[-] HTTP Method '{method}' not allowed.")

    # Test for Directory Traversal
    def test_directory_traversal(self):
        url = f"{self.base_url}/download?file=../../passwd"
        response = requests.get(url)
        if "root:x:" in response.text:
            print("[+] Directory traversal vulnerability detected.")
        else:
            print("[-] Directory traversal not detected.")

    # Test for Weak Password Policies
    def test_weak_password_policies(self):
        weak_passwords = ["123456", "password", "admin", "qwerty"]
        for password in weak_passwords:
            payload = {"username": "admin", "password": password}
            response = requests.post(f"{self.base_url}/login", data=payload)
            if "dashboard" in response.url:
                print(f"[+] Weak password '{password}' allowed.")
            else:
                print(f"[-] Weak password '{password}' not allowed.")

    # Test for SSL/TLS Security (Certificate Validity)
    def test_ssl_tls(self):
        try:
            response = requests.get(self.base_url, verify=True)
            print("[+] SSL/TLS certificate is valid.")
        except requests.exceptions.SSLError:
            print("[-] SSL/TLS certificate validation failed.")

    # Test for Subdomain Takeover
    def test_subdomain_takeover(self):
        subdomains = ["test.website.com", "old.website.com"]
        for subdomain in subdomains:
            try:
                response = requests.get(f"https://{subdomain}")
                if response.status_code == 404:
                    print(f"[+] Subdomain takeover risk detected for {subdomain}.")
                else:
                    print(f"[-] No subdomain takeover risk for {subdomain}.")
            except Exception as e:
                print(f"[-] Error checking {subdomain}: {e}")

    # Cleanup after testing
    def cleanup(self):
        driver.quit()


if __name__ == "__main__":
    tests = PenetrationTests()
    tests.test_sql_injection()
    tests.test_xss()
    tests.test_open_redirect()
    tests.test_brute_force()
    tests.test_ssh_connection()
    tests.test_auth_bypass()
    tests.test_clickjacking()
    tests.test_insecure_http_methods()
    tests.test_directory_traversal()
    tests.test_weak_password_policies()
    tests.test_ssl_tls()
    tests.test_subdomain_takeover()
    tests.cleanup()