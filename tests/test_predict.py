"""
tests/test_predict.py
---------------------
Unit tests for the ML prediction logic.
Run with: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

# ── Sample customer feature vectors ───────────────────────────────────────────
HIGH_RISK_CUSTOMER = {
    "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
    "tenure": 2, "Contract": 0, "PaperlessBilling": 1, "PaymentMethod": 2,
    "MonthlyCharges": 95.0, "TotalCharges": 190.0,
    "PhoneService": 1, "MultipleLines": 0, "InternetService": 1,
    "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0,
    "TechSupport": 0, "StreamingTV": 1, "StreamingMovies": 1,
    "charges_per_tenure": 95.0 / 3, "service_count": 4, "is_long_term": 0,
}

LOW_RISK_CUSTOMER = {
    "SeniorCitizen": 0, "Partner": 1, "Dependents": 1,
    "tenure": 60, "Contract": 2, "PaperlessBilling": 0, "PaymentMethod": 0,
    "MonthlyCharges": 50.0, "TotalCharges": 3000.0,
    "PhoneService": 1, "MultipleLines": 1, "InternetService": 0,
    "OnlineSecurity": 1, "OnlineBackup": 1, "DeviceProtection": 1,
    "TechSupport": 1, "StreamingTV": 0, "StreamingMovies": 0,
    "charges_per_tenure": 50.0 / 61, "service_count": 7, "is_long_term": 1,
}


# ── Test: predict_single returns expected keys ─────────────────────────────────
def test_predict_single_keys():
    from ml.predict import predict_single
    result = predict_single(HIGH_RISK_CUSTOMER)
    assert "probability"  in result
    assert "risk"         in result
    assert "clv_tier"     in result
    assert "priority"     in result
    assert "top_factors"  in result


# ── Test: probability is a float between 0 and 1 ─────────────────────────────
def test_predict_probability_range():
    from ml.predict import predict_single
    for customer in [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]:
        result = predict_single(customer)
        assert 0.0 <= result["probability"] <= 1.0, \
            f"Probability out of range: {result['probability']}"


# ── Test: risk labels are valid ────────────────────────────────────────────────
def test_risk_label_values():
    from ml.predict import predict_single
    valid_labels = {"HIGH", "MEDIUM", "LOW"}
    for customer in [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]:
        result = predict_single(customer)
        assert result["risk"] in valid_labels
        assert result["clv_tier"] in valid_labels


# ── Test: top_factors has at most 5 entries ────────────────────────────────────
def test_top_factors_length():
    from ml.predict import predict_single
    result = predict_single(HIGH_RISK_CUSTOMER)
    assert len(result["top_factors"]) <= 5


# ── Test: each factor has required fields ──────────────────────────────────────
def test_top_factors_structure():
    from ml.predict import predict_single
    result = predict_single(HIGH_RISK_CUSTOMER)
    for factor in result["top_factors"]:
        assert "feature"   in factor
        assert "label"     in factor
        assert "impact"    in factor
        assert "direction" in factor
        assert factor["direction"] in ("increases", "decreases")


# ── Test: recommendations return a list ───────────────────────────────────────
def test_recommendations_list():
    from ml.recommend import get_recommendations
    recs = get_recommendations(HIGH_RISK_CUSTOMER, 0.82, "HIGH", "LOW")
    assert isinstance(recs, list)
    assert len(recs) >= 1
    assert len(recs) <= 3


# ── Test: recommendation has required fields ───────────────────────────────────
def test_recommendation_structure():
    from ml.recommend import get_recommendations
    recs = get_recommendations(HIGH_RISK_CUSTOMER, 0.82, "HIGH", "MEDIUM")
    for rec in recs:
        assert "action"   in rec
        assert "reason"   in rec
        assert "urgency"  in rec
        assert "category" in rec
        assert rec["urgency"] in ("HIGH", "MEDIUM", "LOW")
