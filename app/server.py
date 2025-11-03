# app/server.py
import mlflow
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field
from typing import List

# ---- Hard-coded config (simple, explicit) ----
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME          = "iris-classifier"
MODEL_VERSION       = "1"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(MODEL_URI)

# 11/2 EK: SERVED and _load_version as helpers for homework criteria for dynamic model version call.
SERVED = { #11/2 EK: Addeddictionary to save current model version.
    "version": int(MODEL_VERSION),
    "model": model
}

# 11/2 EK: SERVED and _load_version as helpers for homework criteria for dynamic model version call.

def _load_version(version: int): #11/2 EK Added to update current version.
    """
    Load a specific model version from the MLflow Model Registry
    and set it as the currently served model.
    """
    version = int(version)
    model_uri = f"models:/{MODEL_NAME}/{version}"
    try:
        loaded = mlflow.pyfunc.load_model(model_uri) #https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load {model_uri}: {e}")
    SERVED["version"] = version #Update dictionary value
    SERVED["model"] = loaded #Update dictionary value
    return {
        "model_name": MODEL_NAME,
        "model_uri": model_uri,
        "version": version
    }    

# ----- Pydantic schemas with helpful docs + examples -----
class IrisSample(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width:  float = Field(..., ge=0, description="Petal width in cm")

class PredictRequest(BaseModel):
    samples: List[IrisSample]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "samples": [
                        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                        {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
                        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
                    ]
                }
            ]
        }
    }

# For convenience, return both class ids and human labels
IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

class PredictResponse(BaseModel):
    class_id: List[int]    # 0,1,2
    class_label: List[str] # setosa/versicolor/virginica

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"class_id": [0, 1, 2], "class_label": ["setosa", "versicolor", "virginica"]}
            ]
        }
    }

app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species from sepal/petal measurements (cm).",
    version="1.0.0",
)

# TODO Add endpoint to get the current model serving version
# Homework criterion #6: Provide an endpoint to view the current served version
    
@app.get("/health", tags=["health"], summary="Check API, MLflow connection, and current served model status")
def health():
    return {
        "status": "ok",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_name": MODEL_NAME,
        "served_version": SERVED["version"],
        "model_uri": f"models:/{MODEL_NAME}/{SERVED['version']}",
        "model_is_loaded": SERVED["model"] is not None,
        "endpoints": {
            "serve_model": "/model/serve/{version}",
            "predict": "/predict"
        }
    }

# TODO Add endpoint to update the serving version
# Homework Criterion #5: Create a new API endpoint to allow us to select a version to serve    
@app.post("/model/serve/{version}", tags=["model"], summary="Serve a specific registered model version") #11/2 EK Added to update current version.
def serve_model_version(version: int):
    info = _load_version(version)
    return {
        "status": "ok",
        "message": f"Now serving {info['model_name']} version {info['version']}.",
        **info
    }
    

# TODO Run predict
# TODO Predict using the correct served version
# Homework Criterion #4: Extend the API server to serve this registered model

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Predict Iris species",
    description="Send one or more Iris samples; returns class id (0,1,2) and label (setosa, versicolor, virginica)."
)
def predict(req: PredictRequest) -> PredictResponse:
    if SERVED["model"] is None:
        raise RuntimeError("No model is currently served. Use POST /model/serve/{version} first.")
    
    data = []
    for sample in req.samples:
        data.append([
            sample.sepal_length,
            sample.sepal_width,
            sample.petal_length,
            sample.petal_width
        ])

    df = pd.DataFrame(
        data, 
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )

    preds = SERVED["model"].predict(df)

    class_ids = []
    class_labels = []
    for p in preds:
        class_ids.append(int(p))
        class_labels.append(IRIS_LABELS[int(p)])
    
    return PredictResponse(class_id=class_ids, class_label=class_labels)
    
    
#    preds = model.predict(df)
    

