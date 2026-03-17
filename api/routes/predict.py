"""
api/routes/predict.py
---------------------
POST /api/predict  — single customer churn prediction
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import APIRouter, HTTPException
from api.schemas import CustomerFeatures, PredictionResponse
from ml.predict import predict_single
from ml.recommend import get_recommendations
from ml.segmentation import segment_customer

router = APIRouter()


def _build_feature_dict(data: CustomerFeatures) -> dict:
    """Convert Pydantic model to plain dict and add engineered features."""
    d = data.model_dump()
    d["charges_per_tenure"] = d["MonthlyCharges"] / (d["tenure"] + 1)
    d["service_count"] = sum([
        d.get("PhoneService", 0),
        1 if d.get("MultipleLines", 0) == 1 else 0,
        1 if d.get("InternetService", 0) in [0, 1] else 0,
        1 if d.get("OnlineSecurity", 0) == 1 else 0,
        1 if d.get("OnlineBackup", 0) == 1 else 0,
        1 if d.get("DeviceProtection", 0) == 1 else 0,
        1 if d.get("TechSupport", 0) == 1 else 0,
        1 if d.get("StreamingTV", 0) == 1 else 0,
        1 if d.get("StreamingMovies", 0) == 1 else 0,
    ])
    d["is_long_term"] = int(d["tenure"] > 24)
    return d


@router.post("/predict", response_model=PredictionResponse)
async def predict(data: CustomerFeatures):
    """
    Predict churn probability for a single customer.
    Returns probability, risk tier, SHAP explanations, and recommendations.
    """
    try:
        features = _build_feature_dict(data)
        result   = predict_single(features)
        recs     = get_recommendations(
            features,
            result["probability"],
            result["risk"],
            result["clv_tier"]
        )
        segment = segment_customer(features)

        return PredictionResponse(
            probability=result["probability"],
            risk=result["risk"],
            clv_tier=result["clv_tier"],
            priority=result["priority"],
            top_factors=result["top_factors"],
            recommendations=recs,
            segment=segment,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
