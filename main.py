import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------
# ✅ Initialize FastAPI App
# --------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease risk based on patient data.",
    version="1.0.0",
)

# --------------------------
# ✅ CORS Configuration
# --------------------------
# Read ALLOWED_ORIGINS from environment variable
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS")
if ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS.split(",")]
else:
    # Default: only allow localhost for safety
    ALLOWED_ORIGINS = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# ✅ Load Model, Scaler, Columns
# --------------------------
MODEL_PATH = "knn_heart.pkl"
SCALER_PATH = "scaler.pkl"
COLUMNS_PATH = "columns.pkl"

model, scaler, expected_columns = None, None, None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected_columns = joblib.load(COLUMNS_PATH)
    print("✅ Model, Scaler, and Columns loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading model or artifacts: {e}")

# --------------------------
# ✅ Input Schema
# --------------------------
class HeartInput(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

# --------------------------
# ✅ API Routes
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
# ✅ Prediction Endpoint
# --------------------------
@app.post("/predict")
def predict(data: HeartInput):
    if model is None or scaler is None or expected_columns is None:
        raise HTTPException(
            status_code=500,
            detail="Model or preprocessing artifacts not available.",
        )

    try:
        # Validation for categorical fields
        valid_sex = {"M", "F"}
        valid_chest_pain = {"ATA", "NAP", "TA", "ASY"}
        valid_resting_ecg = {"Normal", "ST", "LVH"}
        valid_exercise_angina = {"Y", "N"}
        valid_st_slope = {"Up", "Flat", "Down"}

        if data.Sex not in valid_sex:
            raise HTTPException(status_code=400, detail="Invalid value for Sex.")
        if data.ChestPainType not in valid_chest_pain:
            raise HTTPException(status_code=400, detail="Invalid value for ChestPainType.")
        if data.RestingECG not in valid_resting_ecg:
            raise HTTPException(status_code=400, detail="Invalid value for RestingECG.")
        if data.ExerciseAngina not in valid_exercise_angina:
            raise HTTPException(status_code=400, detail="Invalid value for ExerciseAngina.")
        if data.ST_Slope not in valid_st_slope:
            raise HTTPException(status_code=400, detail="Invalid value for ST_Slope.")

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

        return {"prediction": result}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed due to server error.")

# --------------------------
# ✅ Run App (Local / Render)
# --------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
