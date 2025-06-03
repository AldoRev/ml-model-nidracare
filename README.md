# Sleep Quality Prediction API

This FastAPI backend serves a TensorFlow.js model for predicting sleep quality based on various health and lifestyle inputs.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your TensorFlow.js model files in the root directory:
   - `model.json` (model architecture)
   - `group1-shard1of1.bin` (model weights)

3. Run the server:
```bash
python main.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### POST /predict
Predicts sleep quality based on input parameters.

Example request:
```json
{
    "age": 25,
    "gender": 1,
    "sleep_duration": 7.5,
    "sleep_quality": 8,
    "bmi_category": "normal",
    "stress_level": 4,
    "physical_activity_duration": 1.5,
    "steps_per_day": 8000
}
```

Example response:
```json
{
    "prediction": [0.1, 0.3, 0.6],
    "predicted_class": "Good",
    "confidence": 0.6
}
```

### GET /model/info
Returns information about the loaded model.

## Input Parameters

- `age`: Age of the person (float)
- `gender`: Gender (0 for female, 1 for male)
- `sleep_duration`: Sleep duration in hours (float)
- `sleep_quality`: Sleep quality rating from 1-10 (integer)
- `bmi_category`: BMI category ("underweight", "normal", or "overweight")
- `stress_level`: Stress level from 1-10 (integer)
- `physical_activity_duration`: Physical activity duration in hours (float)
- `steps_per_day`: Number of steps per day (integer)

## Output Classes

The model predicts one of three sleep quality classes:
- "Poor"
- "Average"
- "Good"

## Features

- RESTful API endpoints for ML model predictions
- Support for both numeric and image input data
- CORS enabled for frontend integration
- Model information endpoint
- Automatic API documentation with Swagger UI

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `