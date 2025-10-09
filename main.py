import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# CORS configuration for production
# In production, set allowed origins to your frontend domain(s)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS")
# ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Set this to your frontend domain(s) in production
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Restrict to only used methods
    allow_headers=["*"],
)

# Load model, scaler, and expected columns
try:
    model = joblib.load("knn_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
except Exception as e:
    # In production, log this error to a file or monitoring system
    print(f"Error loading model or artifacts: {e}")
    model = None
    scaler = None
    expected_columns = None

# Define expected input
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Prediction API"}

@app.post("/predict")
def predict(data: HeartInput):
    # Check if model and artifacts are loaded
    if model is None or scaler is None or expected_columns is None:
        # In production, do not leak internal errors
        raise HTTPException(status_code=500, detail="Model not available. Please try again later.")

    try:
        # Validate categorical values (add more checks as needed)
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


        # Create raw input
        raw_input = {
            'Age': data.Age,
            'RestingBP': data.RestingBP,
            'Cholesterol': data.Cholesterol,
            'FastingBS': data.FastingBS,
            'MaxHR': data.MaxHR,
            'Oldpeak': data.Oldpeak,
            'Sex_' + data.Sex: 1,
            'ChestPainType_' + data.ChestPainType: 1,
            'RestingECG_' + data.RestingECG: 1,
            'ExerciseAngina_' + data.ExerciseAngina: 1,
            'ST_Slope_' + data.ST_Slope: 1
        }

        # Create dataframe
        input_df = pd.DataFrame([raw_input])

        # Add missing columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)[0]

        # Return result
        if prediction == 1:
            return {"prediction": "High Risk of Heart Disease"}
        else:
            return {"prediction": "Low Risk of Heart Disease"}

    except HTTPException as he:
        # Reraise HTTPExceptions for FastAPI to handle
        raise he
    except Exception as e:
        # In production, log the error and return a generic message
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again later.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
