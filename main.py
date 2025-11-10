import os
import joblib
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field, validator
import time

# --------------------------
# Configure Logging
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------
# Initialize FastAPI App
# --------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease risk based on patient data.",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
)

# --------------------------
# Security Middleware
# --------------------------

# Trusted Host Middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.vercel.app", "*.herokuapp.com"]
)

# CORS Configuration - More restrictive
origins = [
    "http://localhost:5173",  # Local React frontend
    "https://heart-disease-predection.vercel.app",  # Deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],  # Restrict methods
    allow_headers=["Content-Type", "Authorization"],  # Restrict headers
    max_age=600,  # Cache preflight for 10 minutes
)

# --------------------------
# Load Model, Scaler, Columns
# --------------------------
MODEL_PATH = "knn_heart_model.pkl"
SCALER_PATH = "heart_scaler.pkl"
COLUMNS_PATH = "heart_columns.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected_columns = joblib.load(COLUMNS_PATH)
    print("✅ Model, Scaler, and Columns loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading model or artifacts: {e}")
    model, scaler, expected_columns = None, None, None

# --------------------------
# Input Schema with Validation
# --------------------------
class HeartInput(BaseModel):
    Age: int = Field(..., ge=0, le=120, description="Age must be between 0 and 120")
    Sex: str = Field(..., regex="^[MF]$", description="Sex must be M or F")
    ChestPainType: str = Field(..., regex="^(ATA|NAP|TA|ASY)$", description="Invalid chest pain type")
    RestingBP: int = Field(..., ge=50, le=250, description="Resting BP must be between 50 and 250")
    Cholesterol: int = Field(..., ge=100, le=600, description="Cholesterol must be between 100 and 600")
    FastingBS: int = Field(..., ge=0, le=1, description="Fasting BS must be 0 or 1")
    RestingECG: str = Field(..., regex="^(Normal|ST|LVH)$", description="Invalid resting ECG type")
    MaxHR: int = Field(..., ge=60, le=220, description="Max HR must be between 60 and 220")
    ExerciseAngina: str = Field(..., regex="^[YN]$", description="Exercise angina must be Y or N")
    Oldpeak: float = Field(..., ge=-5.0, le=10.0, description="Oldpeak must be between -5.0 and 10.0")
    ST_Slope: str = Field(..., regex="^(Up|Flat|Down)$", description="Invalid ST slope type")
    
    @validator('Age')
    def validate_age(cls, v):
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v
    
    @validator('RestingBP')
    def validate_resting_bp(cls, v):
        if v < 50 or v > 250:
            raise ValueError('Resting BP must be between 50 and 250 mmHg')
        return v
    
    @validator('Cholesterol')
    def validate_cholesterol(cls, v):
        if v < 100 or v > 600:
            raise ValueError('Cholesterol must be between 100 and 600 mg/dl')
        return v
    
    @validator('MaxHR')
    def validate_max_hr(cls, v):
        if v < 60 or v > 220:
            raise ValueError('Max HR must be between 60 and 220 bpm')
        return v

# --------------------------
# Request Logging Middleware
# --------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {process_time:.4f}s")
    
    return response

# --------------------------
# API Routes
# --------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Heart Disease Prediction API"}

@app.head("/")
def head_root():
    return Response(status_code=200)

@app.options("/")
def options_root():
    return Response(status_code=200)

# --------------------------
# Prediction Endpoint with Rate Limiting
# --------------------------
@app.post("/predict", dependencies=[RateLimiter(times=10, seconds=60)])  # 10 requests per minute
def predict(data: HeartInput):
    if model is None or scaler is None or expected_columns is None:
        raise HTTPException(
            status_code=500,
            detail="Model or preprocessing artifacts not available.",
        )

    try:
        # Input validation is now handled by Pydantic model
        logger.info(f"Processing prediction request for age {data.Age}, sex {data.Sex}")

        # Create input dict for model
        raw_input = {
            "Age": data.Age,
            "RestingBP": data.RestingBP,
            "Cholesterol": data.Cholesterol,
            "FastingBS": data.FastingBS,
            "MaxHR": data.MaxHR,
            "Oldpeak": data.Oldpeak,
            f"Sex_{data.Sex}": 1,
            f"ChestPainType_{data.ChestPainType}": 1,
            f"RestingECG_{data.RestingECG}": 1,
            f"ExerciseAngina_{data.ExerciseAngina}": 1,
            f"ST_Slope_{data.ST_Slope}": 1,
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([raw_input])

        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]

        # Scale and predict
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
        
        logger.info(f"Prediction completed: {result}")
        return {"prediction": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed due to server error.")

# --------------------------
# Initialize Rate Limiter
# --------------------------
@app.on_event("startup")
async def startup():
    # Initialize rate limiter (in production, use Redis)
    await FastAPILimiter.init()

# --------------------------
# Run App
# --------------------------
if __name__ == "__main__":
    import uvicorn
    
    # Validate environment variables
    port = int(os.environ.get("PORT", 8000))
    environment = os.environ.get("ENVIRONMENT", "development")
    
    logger.info(f"Starting server in {environment} mode on port {port}")
    
    # Security: Don't use reload in production
    reload = environment == "development"
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=reload,
        access_log=True,
        log_level="info"
    )
