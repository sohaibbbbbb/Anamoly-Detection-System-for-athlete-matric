import joblib
import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Import your existing neural network class
from anamoly_detection_nn import AnomalyDetectorANN

# 1. Define the Expected JSON Payload (Matching C++ Struct)
class AthleteMetricInput(BaseModel):
    athlete_id: int
    bodyweight: float
    squat: float
    bench: float
    deadlift: float

app = FastAPI(title="Iron-Logic Anti-Doping API")

# Global variables for models
preprocessor = None
nn_model = None

@app.on_event("startup")
def load_system_binaries() -> None:
    """Loads the preprocessing pipeline and neural network into memory on boot."""
    global preprocessor, nn_model
    try:
        # 1. Load Scikit-Learn Preprocessor
        preprocessor = joblib.load("production_pipeline.pkl")
        
        # 2. Load Saved Neural Network Weights
        state_dict = torch.load("anomaly_weights.pth")
        
        # 3. Dynamically inspect tensor shape [16, input_dim] from weights
        dynamic_input_dim = state_dict['layer_1.weight'].shape[1]
        
        # 4. Instantiate Model with Matching Input Dimension
        nn_model = AnomalyDetectorANN(input_dim=dynamic_input_dim, hidden_dim=16)
        nn_model.load_state_dict(state_dict)
        nn_model.eval() # Set to evaluation mode for inference
        
    except FileNotFoundError as e:
        raise RuntimeError(f"Critical System Failure: Missing binary file. {e}")
    
@app.post("/predict")
def predict_anomaly(metric: AthleteMetricInput) -> Dict[str, Any]:
    """Ingests raw metrics, preprocesses via Joblib, and infers via PyTorch."""
    if preprocessor is None or nn_model is None:
        raise HTTPException(status_code=500, detail="Models failed to load.")

    # 1. Convert JSON payload to Pandas DataFrame
    input_df = pd.DataFrame([metric.dict()])
    
    # 2. Preprocess (Transform raw numbers to scaled mathematical matrix)
    try:
        X_processed = preprocessor.transform(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failure: {e}")

    # 3. Tensor Casting & Inference
    input_tensor = torch.tensor(X_processed, dtype=torch.float32)
    
    with torch.no_grad(): # Disable calculus engine for speed
        prediction_prob = nn_model(input_tensor).item()
        
    # 4. Binary Classification Threshold
    is_anomaly = bool(prediction_prob >= 0.5)

    return {
        "athlete_id": metric.athlete_id,
        "anomaly_probability": round(prediction_prob, 4),
        "flagged": is_anomaly
    }