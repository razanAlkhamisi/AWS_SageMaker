# inference.py
import joblib
import json
import os
import numpy as np

# Define the exact feature order used for training
FEATURE_ORDER = [
    "age", "bmi", "glucose", "blood_pressure", "insulin",
    "skin_thickness", "pregnancies", "diabetes_pedigree",
    "feature9", "feature10"
]

def model_fn(model_dir):
    """Load trained model from S3 bundle"""
    model_path = os.path.join(model_dir, "diabetes_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse JSON input"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        if not isinstance(data, dict):
            raise ValueError("Input JSON must be a dictionary of features")
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Convert input dict to 2D array and predict"""
    # Ensure all features exist
    missing = [feat for feat in FEATURE_ORDER if feat not in input_data]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Convert dict to ordered numeric array
    X = np.array([input_data[feat] for feat in FEATURE_ORDER], dtype=float).reshape(1, -1)
    prediction = model.predict(X)
    return prediction.tolist()

def output_fn(prediction, accept):
    """Return JSON output"""
    if accept == "application/json":
        return json.dumps({"prediction": prediction}), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
