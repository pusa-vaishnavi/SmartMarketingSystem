
from pathlib import Path
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load models from the models folder
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

kmeans = joblib.load(MODEL_DIR / "kmeans_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
pca = joblib.load(MODEL_DIR / "pca_model.pkl")
cluster_name_map = joblib.load(
    MODEL_DIR / "cluster_name_map.pkl"
)

# Create FastAPI application
app = FastAPI(
    title="Smart Marketing Decision Support API",
    description="Customer Segmentation using RFM and K-Means",
    version="1.0"
)

# Input data
class CustomerData(BaseModel):
    Recency: float = Field(default=10, ge=0)
    Frequency: float = Field(default=15, ge=1)
    Monetary: float = Field(default=5000, gt=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Recency": 10,
                "Frequency": 15,
                "Monetary": 5000
            }
        }
    }

@app.get("/")
def home():
    return {
        "message": "Smart Marketing API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "K-Means Customer Segmentation"
    }


@app.post("/predict")
def predict_customer(data: CustomerData):

    try:
        features = np.array([[
            data.Recency,
            data.Frequency,
            data.Monetary
        ]])

        scaled_features = scaler.transform(features)

        cluster = int(kmeans.predict(scaled_features)[0])

        segment = cluster_name_map.get(
            cluster,
            cluster_name_map.get(str(cluster), "Unknown Segment")
        )

        pca_values = pca.transform(scaled_features)

        return {
            "Customer_RFM": {
                "Recency": data.Recency,
                "Frequency": data.Frequency,
                "Monetary": data.Monetary
            },
            "Predicted_Cluster": cluster,
            "Customer_Segment": segment,
            "PCA1": round(float(pca_values[0][0]), 3),
            "PCA2": round(float(pca_values[0][1]), 3)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
