import os
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Prediction

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set.")

engine = create_async_engine(DATABASE_URL, echo=True, future=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

app = FastAPI()

# Dependency for DB session
async def get_db():
    async with async_session() as session:
        yield session

# Pydantic input model
class PredictInput(BaseModel):
    user_id: int
    age: float
    gender: int
    sleep_duration: float
    sleep_quality: int
    bmi_category: str
    stress_level: int
    physical_activity_duration: float
    steps_per_day: int

# --- PATCHING FUNCTIONS FOR TFJS -> KERAS COMPATIBILITY ---

def patch_layer_config(layer_config):
    # Patch dtype
    if "dtype" in layer_config and isinstance(layer_config["dtype"], dict):
        if "config" in layer_config["dtype"] and "name" in layer_config["dtype"]["config"]:
            layer_config["dtype"] = layer_config["dtype"]["config"]["name"]
        else:
            layer_config["dtype"] = "float32"
    # Patch kernel_initializer and bias_initializer and others
    for key in [
        "kernel_initializer", "bias_initializer",
        "beta_initializer", "gamma_initializer",
        "moving_mean_initializer", "moving_variance_initializer"
    ]:
        if key in layer_config and isinstance(layer_config[key], dict):
            if "class_name" in layer_config[key]:
                layer_config[key] = layer_config[key]["class_name"]
            else:
                # Default fallback
                layer_config[key] = "Zeros" if "bias" in key or "beta" in key or "moving_mean" in key else "Ones"
    return layer_config

def patch_model_config(model_config):
    # Patch InputLayer config for Keras compatibility and all layers for dtype/initializers
    if "layers" in model_config["config"]:
        for layer in model_config["config"]["layers"]:
            if layer["class_name"] == "InputLayer":
                config = layer["config"]
                if "batch_shape" in config:
                    config["batch_input_shape"] = config.pop("batch_shape")
            # Patch all layers for dtype and initializers
            layer["config"] = patch_layer_config(layer["config"])
    return model_config

# --- LOAD MODEL AT STARTUP ---
model = None

def load_tfjs_model():
    try:
        with open("model.json", "r") as f:
            model_json = json.load(f)
        model_config = model_json["modelTopology"]["model_config"]
        model_config = patch_model_config(model_config)
        model = tf.keras.models.model_from_json(json.dumps(model_config))
        weights_dict = np.load("model_weights.npy", allow_pickle=True).item()
        # Map TFJS weight names to Keras layer weights
        for layer in model.layers:
            layer_weights = []
            for weight_tensor in layer.weights:
                # Keras weight name: e.g. 'dense_3/kernel:0' or 'dense_3/bias:0'
                # TFJS weight name: e.g. 'sequential_1/dense_3/kernel'
                keras_name = weight_tensor.name
                # Extract layer and weight type
                # e.g. 'dense_3/kernel:0' -> 'dense_3/kernel'
                keras_name = keras_name.replace(":0", "")
                # Find the matching TFJS weight name
                for tfjs_name in weights_dict:
                    if keras_name in tfjs_name:
                        layer_weights.append(weights_dict[tfjs_name])
                        break
                else:
                    # If not found, append zeros
                    layer_weights.append(np.zeros(weight_tensor.shape))
            if layer_weights:
                layer.set_weights(layer_weights)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

@app.on_event("startup")
async def startup_event():
    global model
    model = load_tfjs_model()
    if model is None:
        print("Warning: Model failed to load")
    else:
        print("Model input shape:", model.input_shape)
        print("Model output shape:", model.output_shape)

@app.post("/predict")
async def predict(input: PredictInput, db: AsyncSession = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    # Preprocess input
    bmi_map = {"underweight": 0, "normal": 1, "overweight": 2}
    try:
        bmi_val = bmi_map[input.bmi_category.lower()]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid BMI category")
    features = np.array([[input.age, input.gender, input.sleep_duration, input.sleep_quality,
                          bmi_val, input.stress_level, input.physical_activity_duration, input.steps_per_day]])
    # Pad to 11 features if needed
    if features.shape[1] < 11:
        features = np.concatenate([features, np.zeros((1, 11 - features.shape[1]))], axis=1)
    # Predict
    pred = model.predict(features)
    pred = pred[0]
    classes = ["Poor", "Average", "Good"]
    predicted_class = classes[int(np.argmax(pred))]
    confidence = float(np.max(pred))
    # Save to DB
    new_pred = Prediction(
        user_id=input.user_id,
        age=input.age,
        gender=input.gender,
        sleep_duration=input.sleep_duration,
        sleep_quality=input.sleep_quality,
        bmi_category=input.bmi_category,
        stress_level=input.stress_level,
        physical_activity_duration=input.physical_activity_duration,
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

# Utility: create tables (run once, then comment/remove)
# if __name__ == "__main__":
#     import asyncio
#     async def create_tables():
#         async with engine.begin() as conn:
#             await conn.run_sync(Base.metadata.create_all)
#     asyncio.run(create_tables())