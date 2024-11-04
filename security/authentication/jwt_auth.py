import jwt
import datetime
from functools import wraps
from flask import request, jsonify, abort
from werkzeug.security import generate_password_hash, check_password_hash

# Secret key for encoding and decoding JWTs
SECRET_KEY = 'SecretKey'

# User database 
users_db = {
    "person1": {
        "username": "person1",
        "password": generate_password_hash("password1", method='sha256'),
        "role": "admin"
    },
    "person2": {
        "username": "person2",
        "password": generate_password_hash("password2", method='sha256'),
        "role": "user"
    }
}

# Initialize Flask app
app = Flask(__name__)

# Function to generate a JWT token
def generate_token(username):
    expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    payload = {
        'username': username,
        'exp': expiration
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

# Decorator for verifying the JWT token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # Check for token in request headers
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            # Decode the token
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = users_db.get(data['username'])
            if not current_user:
                raise ValueError('User not found')
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401
        except Exception as e:
            return jsonify({'message': str(e)}), 401

        return f(current_user, *args, **kwargs)
    
    return decorated

# Login endpoint for user authentication
def login():
    auth = request.authorization

    if not auth or not auth.username or not auth.password:
        return jsonify({'message': 'Could not verify'}), 401

    user = users_db.get(auth.username)
    
    if not user:
        return jsonify({'message': 'User not found'}), 401

    if check_password_hash(user['password'], auth.password):
        token = generate_token(auth.username)
        return jsonify({'token': token})

    return jsonify({'message': 'Password is incorrect'}), 401

# Protected route
@token_required
def protected(current_user):
    return jsonify({
        'message': f'Hello, {current_user["username"]}! You are an {current_user["role"]}.'
    })

# Role-based access control (RBAC) decorator
def role_required(required_role):
    def wrapper(f):
        @wraps(f)
        @token_required
        def decorated(current_user, *args, **kwargs):
            if current_user['role'] != required_role:
                return jsonify({'message': 'Access Denied: Insufficient Permissions'}), 403
            return f(current_user, *args, **kwargs)
        return decorated
    return wrapper

# Admin-only route
@role_required('admin')
def admin_route(current_user):
    return jsonify({
        'message': f'Hello, {current_user["username"]}. You have admin access!'
    })

# Error handling
@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'message': 'Unauthorized access'}), 401

@app.errorhandler(403)
def forbidden(error):
    return jsonify({'message': 'Forbidden access'}), 403

@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Resource not found'}), 404

# Flask app and route setup
from flask import Flask

app = Flask(__name__)

# Routes
@app.route('/login', methods=['POST'])
def login_route():
    return login()

@app.route('/protected', methods=['GET'])
@token_required
def protected_route(current_user):
    return protected(current_user)

@app.route('/admin', methods=['GET'])
@role_required('admin')
def admin_route_handler(current_user):
    return admin_route(current_user)

if __name__ == '__main__':
    app.run(debug=True)