import os
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Prediction
import joblib

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set.")

engine = create_async_engine(DATABASE_URL, echo=True, future=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

async def get_db():
    async with async_session() as session:
        yield session

class PredictInput(BaseModel):
    user_id: int
    gender: str
    age: int
    sleep_duration: float
    sleep_quality: int
    physical_activity_duration: int
    stress_level: int
    bmi_category: str
    steps_per_day: int
    
# MODEL & SCALER
model = None
scaler = None

def load_keras_model():
    try:
        model = tf.keras.models.load_model("sleep_disorder_prediction_model.h5")
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def load_scaler():
    try:
        scaler = joblib.load("scaler.save")
        print("Scaler loaded successfully!")
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {str(e)}")
        return None

@app.on_event("startup")
async def startup_event():
    global model, scaler
    model = load_keras_model()
    scaler = load_scaler()
    if model is None:
        print("Warning: Model failed to load")
    else:
        print("Model input shape:", model.input_shape)
        print("Model output shape:", model.output_shape)
    if scaler is None:
        print("Warning: Scaler failed to load")

@app.post("/predict")
async def predict(input: PredictInput, db: AsyncSession = Depends(get_db)):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    gender_map = {"male": 1, "female": 2}
    bmi_map = {"normal": 0, "normal weight": 1, "obese": 2, "overweight": 3}
    try:
        gender_val = gender_map[input.gender.strip().lower()]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid gender (must be 'Male' or 'Female')")
    try:
        bmi_val = bmi_map[input.bmi_category.strip().lower()]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid BMI category")
    features = np.array([[
        gender_val,                          
        input.age,                           
        input.sleep_duration,                
        input.sleep_quality,                 
        input.physical_activity_duration,    
        input.stress_level,                  
        bmi_val,                             
        input.steps_per_day                  
    ]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)
    pred = pred[0]
    classes = ["Normal", "Sleep Apnea", "Insomnia"]  
    predicted_class = classes[int(np.argmax(pred))]
    confidence = float(np.max(pred))
    new_pred = Prediction(
        user_id=input.user_id,
        gender=input.gender.strip().lower(),
        age=input.age,
        sleep_duration=input.sleep_duration,
        sleep_quality=input.sleep_quality,
        physical_activity_duration=input.physical_activity_duration,
        stress_level=input.stress_level,
        bmi_category=input.bmi_category,
        steps_per_day=input.steps_per_day,
        
        prediction=pred.tolist(),
        predicted_class=predicted_class,
        confidence=confidence
    )
    db.add(new_pred)
    await db.commit()
    return {
        "prediction": pred.tolist(),
        "predicted_class": predicted_class,
        "confidence": confidence
    }

@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "layers": [layer.name for layer in model.layers]
    }

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Create prediction table 
# if __name__ == "__main__":
#     import asyncio
#     async def create_tables():
#         async with engine.begin() as conn:
#             await conn.run_sync(Base.metadata.create_all)
#     asyncio.run(create_tables())