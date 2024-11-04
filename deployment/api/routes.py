from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
from sklearn.preprocessing import StandardScaler
import jwt
from typing import List
import logging

# JWT configuration and secret key
SECRET_KEY = "secret_key_here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize FastAPI application
app = FastAPI()

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained model
with open("models/saved_models/best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load scaler for feature normalization
with open("features/scaling_normalization.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Pydantic model for request body
class ChurnPredictionInput(BaseModel):
    customer_id: int
    features: List[float]

# Pydantic model for prediction response
class PredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float
    will_churn: bool

# Model info response
class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    accuracy: float

# Authentication and user verification
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(token: str = Depends(oauth2_scheme)):
    username = verify_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: ChurnPredictionInput, token: str = Depends(get_current_user)):
    logger.info(f"Received request for customer {data.customer_id}")
    
    # Normalize features
    scaled_features = scaler.transform([data.features])
    
    # Predict churn probability
    churn_probability = model.predict_proba(scaled_features)[0][1]
    will_churn = churn_probability > 0.5
    
    # Log prediction
    logger.info(f"Prediction for customer {data.customer_id}: {churn_probability}")
    
    # Return response
    return PredictionResponse(
        customer_id=data.customer_id,
        churn_probability=churn_probability,
        will_churn=will_churn
    )

# Model info endpoint
@app.get("/model_info", response_model=ModelInfoResponse)
def get_model_info(token: str = Depends(get_current_user)):
    # Model details
    model_name = "Churn Prediction Model"
    version = "1.0"
    accuracy = 0.92
    
    # Log model info request
    logger.info("Model information requested")
    
    return ModelInfoResponse(
        model_name=model_name,
        version=version,
        accuracy=accuracy
    )

# Health check endpoint
@app.get("/health")
def health_check():
    logger.info("Health check requested")
    return {"status": "API is healthy"}

# Token generation for login
@app.post("/token")
def login():
    token_data = {"sub": "user"}
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

# Error handling middleware
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request URL: {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Exception handler
@app.exception_handler(Exception)
async def unicorn_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "Internal server error"},
    )