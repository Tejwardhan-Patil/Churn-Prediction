import jwt
import datetime
from functools import wraps
from flask import Flask, request, jsonify, abort

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisissecret'

# Roles and permissions
roles_permissions = {
    'admin': ['create', 'read', 'update', 'delete'],
    'editor': ['create', 'read', 'update'],
    'viewer': ['read']
}

# Users data
users_db = {
    'Person1': {'role': 'admin'},
    'Person2': {'role': 'editor'},
    'Person3': {'role': 'viewer'}
}

# Function to generate JWT tokens
def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    }
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    return token

# Function to verify JWT tokens
def verify_token(token):
    try:
        decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return decoded['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# Function to check if user has required permission
def has_permission(user_id, permission):
    role = users_db.get(user_id, {}).get('role', None)
    if role and permission in roles_permissions.get(role, []):
        return True
    return False

# Decorator to protect routes
def require_permission(permission):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                abort(403, 'Token is missing!')
            
            user_id = verify_token(token)
            if not user_id:
                abort(403, 'Invalid or expired token!')
            
            if not has_permission(user_id, permission):
                abort(403, 'Permission denied!')

            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Routes with different permissions
@app.route('/data/create', methods=['POST'])
@require_permission('create')
def create_data():
    return jsonify({'message': 'Data created successfully!'})

@app.route('/data/read', methods=['GET'])
@require_permission('read')
def read_data():
    return jsonify({'message': 'Data read successfully!'})

@app.route('/data/update', methods=['PUT'])
@require_permission('update')
def update_data():
    return jsonify({'message': 'Data updated successfully!'})

@app.route('/data/delete', methods=['DELETE'])
@require_permission('delete')
def delete_data():
    return jsonify({'message': 'Data deleted successfully!'})

# Route to authenticate users and return a token
@app.route('/auth/login', methods=['POST'])
def login():
    auth = request.json
    if 'user_id' not in auth:
        abort(403, 'User ID is required!')
    
    user_id = auth['user_id']
    if user_id not in users_db:
        abort(403, 'User does not exist!')

    token = generate_token(user_id)
    return jsonify({'token': token})

# Error handler for 403
@app.errorhandler(403)
def forbidden(e):
    return jsonify(error=str(e)), 403

# Error handler for 404
@app.errorhandler(404)
def not_found(e):
    return jsonify(error='Resource not found!'), 404

# Main entry point to run the app
if __name__ == '__main__':
    app.run(debug=True)