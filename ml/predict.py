"""
ml/predict.py
-------------
Core prediction logic used by the API.
Handles: probability scoring, SHAP explanations, CLV classification.
"""

import pickle
import numpy as np
import shap

# ── Load Artifacts (once at import time) ──────────────────────────────────────
MODEL_PATH    = "ml/model_artifacts/xgb_model.pkl"
SCALER_PATH   = "ml/model_artifacts/scaler.pkl"
ENCODERS_PATH = "ml/model_artifacts/label_encoders.pkl"
FEATURES_PATH = "ml/model_artifacts/feature_names.pkl"

model         = pickle.load(open(MODEL_PATH,    "rb"))
scaler        = pickle.load(open(SCALER_PATH,   "rb"))
label_encoders = pickle.load(open(ENCODERS_PATH, "rb"))
feature_names  = pickle.load(open(FEATURES_PATH, "rb"))

# SHAP explainer (TreeExplainer is fast — no GPU needed)
explainer = shap.TreeExplainer(model)

# Human-readable labels for features shown in the UI
FEATURE_LABELS = {
    "tenure":              "Customer tenure (months)",
    "MonthlyCharges":      "Monthly charges",
    "TotalCharges":        "Total charges paid",
    "Contract":            "Contract type",
    "PaymentMethod":       "Payment method",
    "InternetService":     "Internet service type",
    "charges_per_tenure":  "Charges relative to tenure",
    "service_count":       "Number of services subscribed",
    "is_long_term":        "Long-term customer flag",
    "TechSupport":         "Has tech support",
    "OnlineSecurity":      "Has online security",
    "PaperlessBilling":    "Paperless billing",
    "SeniorCitizen":       "Senior citizen",
    "Partner":             "Has partner",
    "Dependents":          "Has dependents",
}


def get_risk_label(probability: float) -> str:
    if probability >= 0.65:
        return "HIGH"
    elif probability >= 0.35:
        return "MEDIUM"
    return "LOW"


def get_clv_tier(total_charges: float) -> str:
    if total_charges > 4000:
        return "HIGH"
    elif total_charges > 1500:
        return "MEDIUM"
    return "LOW"


def get_priority(risk: str, clv: str) -> str:
    matrix = {
        ("HIGH",   "HIGH"):   "CRITICAL",
        ("HIGH",   "MEDIUM"): "HIGH",
        ("HIGH",   "LOW"):    "MEDIUM",
        ("MEDIUM", "HIGH"):   "HIGH",
        ("MEDIUM", "MEDIUM"): "MEDIUM",
        ("MEDIUM", "LOW"):    "LOW",
        ("LOW",    "HIGH"):   "MEDIUM",
        ("LOW",    "MEDIUM"): "LOW",
        ("LOW",    "LOW"):    "LOW",
    }
    return matrix.get((risk, clv), "MEDIUM")


def predict_single(features: dict) -> dict:
    """
    Takes a dict of raw customer features and returns
    probability, risk, SHAP top factors, CLV tier, and priority.
    """
    # Build feature vector in the same order as training
    row = np.array([[features.get(f, 0) for f in feature_names]])

    # Predict
    probability = float(model.predict_proba(row)[0][1])
    risk        = get_risk_label(probability)
    clv_tier    = get_clv_tier(features.get("TotalCharges", 0))
    priority    = get_priority(risk, clv_tier)

    # SHAP explanation
    shap_values = explainer.shap_values(row)
    # For binary XGBoost, shap_values may be 2D or 3D — normalise
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    top_factors = sorted(
        [
            {
                "feature": f,
                "label":   FEATURE_LABELS.get(f, f),
                "impact":  round(float(v), 4),
                "direction": "increases" if v > 0 else "decreases",
            }
            for f, v in zip(feature_names, sv)
        ],
        key=lambda x: abs(x["impact"]),
        reverse=True
    )[:5]

    return {
        "probability": round(probability, 4),
        "risk":        risk,
        "clv_tier":    clv_tier,
        "priority":    priority,
        "top_factors": top_factors,
    }
