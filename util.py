from flask import request, jsonify
import jwt
from functools import wraps

def verifyToken(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers['authorization']
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403
        try:
            print(token.split(' ')[1])
            data = jwt.decode(token.split(' ')[1], 'RESTFULAPIs')
            print(data)
        except:
            return jsonify({'message': 'Token is invalid!'}), 403
        return f(*args, **kwargs)
    return decorated
