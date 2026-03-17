"""
api/routes/simulate.py
----------------------
POST /api/simulate  — What-If simulator
Compare churn probability between original and modified customer attributes.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import APIRouter, HTTPException
from api.schemas import SimulateRequest, SimulateResponse
from ml.predict import predict_single

router = APIRouter()


def _build_feature_dict(data) -> dict:
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


@router.post("/simulate", response_model=SimulateResponse)
async def simulate(data: SimulateRequest):
    """
    Compare churn probability between two customer scenarios.
    Shows the impact of changing a single attribute (e.g. contract type).
    """
    try:
        orig_features = _build_feature_dict(data.original)
        mod_features  = _build_feature_dict(data.modified)

        orig_result = predict_single(orig_features)
        mod_result  = predict_single(mod_features)

        delta = round(mod_result["probability"] - orig_result["probability"], 4)

        return SimulateResponse(
            original_probability=orig_result["probability"],
            modified_probability=mod_result["probability"],
            probability_delta=delta,
            original_risk=orig_result["risk"],
            modified_risk=mod_result["risk"],
            risk_changed=orig_result["risk"] != mod_result["risk"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
