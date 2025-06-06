---
title: Sleep Disorder Prediction API
emoji: 💤
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# Sleep Disorder Prediction API

A FastAPI backend serving a TensorFlow model for predicting sleep disorders...

## 🛣️ API Endpoints

### `POST /predict`
Predicts sleep disorder class from input features.

**Request JSON:**
```json
{
    "user_id": 1,
    "gender": "female",
    "age": 22,
    "sleep_duration": 8.0,
    "physical_activity_duration": 90,
    "stress_level": 2,
    "bmi_category": "normal",
    "steps_per_day": 12000,
    "sleep_quality": 9
}
```

**Response JSON:**
```json
{
  "prediction": [0.1, 0.3, 0.6],
  "predicted_class": "Insomnia",
  "confidence": 0.6
}
```

---

### `GET /model/info`
Returns information about the loaded model.

**Response:**
```json
{
  "input_shape": [null, 8],
  "output_shape": [null, 3],
  "layers": ["dense", "dense_1", ...]
}
```

---

## 📥 Input Parameters

- `user_id`: User ID (int)
- `gender`: "male" or "female"
- `age`: Age (int)
- `sleep_duration`: Sleep duration in hours (float)
- `sleep_quality`: Sleep quality (int)
- `physical_activity_duration`: Physical activity duration in minutes (int)
- `stress_level`: Stress level (int)
- `bmi_category`: "normal", "normal weight", "obese", or "overweight"
- `steps_per_day`: Steps per day (int)

---

## 🏷️ Output Classes

- `"Normal"`
- `"Sleep Apnea"`
- `"Insomnia"`

## 🗄️ Database

- Uses PostgreSQL (asyncpg) for storing predictions.
- Tables: `users`, `predictions` (see `models.py`).

---

## 🧩 Model & Scaler Files

- `sleep_disorder_prediction_model.h5` — Keras model file
- `scaler.save` — Preprocessing scaler (joblib)