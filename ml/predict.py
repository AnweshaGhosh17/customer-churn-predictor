"""
ml/predict.py
-------------
Core prediction logic used by the API.
Handles: probability scoring, SHAP explanations, CLV classification.

FIXES vs original:
  - Lazy loading: models are loaded on first call, not at import time.
    This means the API starts up even when .pkl files don't exist yet.
  - Absolute paths: uses __file__ so paths work on Vercel and locally.
  - Graceful fallback: if models are missing, returns a sensible mock
    response so the frontend can still be demoed end-to-end.
"""

import os
import pickle
import numpy as np

# ── Absolute paths (works on Vercel, locally, anywhere) ────────────────────────
_ML_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(_ML_DIR, "model_artifacts", "xgb_model.pkl")
SCALER_PATH   = os.path.join(_ML_DIR, "model_artifacts", "scaler.pkl")
ENCODERS_PATH = os.path.join(_ML_DIR, "model_artifacts", "label_encoders.pkl")
FEATURES_PATH = os.path.join(_ML_DIR, "model_artifacts", "feature_names.pkl")

# ── Lazy-loaded globals (populated on first predict call) ──────────────────────
_model         = None
_scaler        = None
_label_encoders = None
_feature_names  = None
_explainer      = None
_models_loaded  = False
_models_missing = False   # True once we've confirmed files aren't there yet

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

# Fallback feature order used when feature_names.pkl is missing
_FALLBACK_FEATURES = [
    "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
    "charges_per_tenure", "service_count", "is_long_term",
]


def _try_load_models():
    """
    Attempt to load model artifacts.
    Called lazily on first predict request.
    Sets _models_loaded = True on success, _models_missing = True on failure.
    """
    global _model, _scaler, _label_encoders, _feature_names
    global _explainer, _models_loaded, _models_missing

    if _models_loaded or _models_missing:
        return  # already attempted

    try:
        import shap
        _model          = pickle.load(open(MODEL_PATH,    "rb"))
        _scaler         = pickle.load(open(SCALER_PATH,   "rb"))
        _label_encoders = pickle.load(open(ENCODERS_PATH, "rb"))
        _feature_names  = pickle.load(open(FEATURES_PATH, "rb"))
        _explainer      = shap.TreeExplainer(_model)
        _models_loaded  = True
        print("[predict] Model artifacts loaded successfully.")
    except FileNotFoundError as e:
        _models_missing = True
        print(f"[predict] WARNING: Model artifacts not found ({e}). "
              "Running in demo/mock mode — run notebooks/03_model_training.ipynb first.")
    except Exception as e:
        _models_missing = True
        print(f"[predict] ERROR loading models: {e}. Falling back to mock mode.")


# ── Helper functions ───────────────────────────────────────────────────────────

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


def _mock_predict(features: dict) -> dict:
    """
    Deterministic mock prediction used when model artifacts aren't available.
    Based purely on business rules so the frontend can be demoed.
    """
    # Simple rule-based score so results feel realistic
    score = 0.3
    if features.get("Contract", 0) == 0:
        score += 0.2
    if features.get("tenure", 12) < 12:
        score += 0.15
    if features.get("MonthlyCharges", 65) > 75:
        score += 0.1
    if features.get("TechSupport", 0) == 0:
        score += 0.05
    if features.get("PaymentMethod", 0) == 2:
        score += 0.05
    probability = min(round(score, 4), 0.99)

    risk     = get_risk_label(probability)
    clv_tier = get_clv_tier(features.get("TotalCharges", 0))

    top_factors = [
        {"feature": "Contract",       "label": "Contract type",           "impact":  0.18, "direction": "increases"},
        {"feature": "tenure",         "label": "Customer tenure (months)","impact": -0.12, "direction": "decreases"},
        {"feature": "MonthlyCharges", "label": "Monthly charges",         "impact":  0.09, "direction": "increases"},
        {"feature": "TechSupport",    "label": "Has tech support",        "impact": -0.07, "direction": "decreases"},
        {"feature": "PaymentMethod",  "label": "Payment method",          "impact":  0.05, "direction": "increases"},
    ]

    return {
        "probability": probability,
        "risk":        risk,
        "clv_tier":    clv_tier,
        "priority":    get_priority(risk, clv_tier),
        "top_factors": top_factors,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def predict_single(features: dict) -> dict:
    """
    Takes a dict of raw customer features and returns
    probability, risk, SHAP top factors, CLV tier, and priority.

    Falls back to rule-based mock if model artifacts are missing.
    """
    _try_load_models()

    if _models_missing:
        return _mock_predict(features)

    feature_names = _feature_names or _FALLBACK_FEATURES

    # Build feature vector in the same order as training
    row = np.array([[features.get(f, 0) for f in feature_names]])

    # Predict
    probability = float(_model.predict_proba(row)[0][1])
    risk        = get_risk_label(probability)
    clv_tier    = get_clv_tier(features.get("TotalCharges", 0))
    priority    = get_priority(risk, clv_tier)

    # SHAP explanation
    shap_values = _explainer.shap_values(row)
    # For binary XGBoost, shap_values may be 2D or 3D — normalise
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    top_factors = sorted(
        [
            {
                "feature":   f,
                "label":     FEATURE_LABELS.get(f, f),
                "impact":    round(float(v), 4),
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
