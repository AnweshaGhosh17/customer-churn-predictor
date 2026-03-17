"""
tests/test_api.py
-----------------
Integration tests for FastAPI endpoints.
Run with: pytest tests/test_api.py
Requires the model artifacts to be present.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from api.index import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
    "tenure": 8, "Contract": 0, "PaperlessBilling": 1,
    "PaymentMethod": 2, "MonthlyCharges": 80.0, "TotalCharges": 640.0,
    "PhoneService": 1, "MultipleLines": 0, "InternetService": 1,
    "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0,
    "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0,
}


# ── Health check ───────────────────────────────────────────────────────────────
def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# ── Root endpoint ──────────────────────────────────────────────────────────────
def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "endpoints" in res.json()


# ── Predict endpoint ───────────────────────────────────────────────────────────
def test_predict_status():
    res = client.post("/api/predict", json=SAMPLE_PAYLOAD)
    assert res.status_code == 200


def test_predict_response_shape():
    res  = client.post("/api/predict", json=SAMPLE_PAYLOAD)
    data = res.json()
    assert "probability"     in data
    assert "risk"            in data
    assert "clv_tier"        in data
    assert "top_factors"     in data
    assert "recommendations" in data


def test_predict_probability_range():
    res  = client.post("/api/predict", json=SAMPLE_PAYLOAD)
    data = res.json()
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_invalid_payload():
    res = client.post("/api/predict", json={"tenure": "not_a_number"})
    assert res.status_code == 422   # Pydantic validation error


# ── Simulate endpoint ──────────────────────────────────────────────────────────
def test_simulate_status():
    payload = {"original": SAMPLE_PAYLOAD, "modified": {**SAMPLE_PAYLOAD, "Contract": 1}}
    res = client.post("/api/simulate", json=payload)
    assert res.status_code == 200


def test_simulate_response_shape():
    payload = {"original": SAMPLE_PAYLOAD, "modified": {**SAMPLE_PAYLOAD, "Contract": 2}}
    data    = client.post("/api/simulate", json=payload).json()
    assert "original_probability" in data
    assert "modified_probability" in data
    assert "probability_delta"    in data
    assert "risk_changed"         in data


def test_simulate_contract_upgrade_lowers_risk():
    """Upgrading from month-to-month to 2-year contract should reduce churn probability."""
    original = {**SAMPLE_PAYLOAD, "Contract": 0}
    modified = {**SAMPLE_PAYLOAD, "Contract": 2}
    data = client.post("/api/simulate", json={"original": original, "modified": modified}).json()
    assert data["probability_delta"] <= 0, \
        "Contract upgrade should reduce or maintain churn probability"


# ── Dashboard endpoint ─────────────────────────────────────────────────────────
def test_dashboard_status():
    res = client.get("/api/dashboard")
    assert res.status_code == 200


def test_dashboard_response_shape():
    data = client.get("/api/dashboard").json()
    assert "total_customers"   in data
    assert "high_risk_count"   in data
    assert "top_churn_drivers" in data


def test_dashboard_customers_status():
    res = client.get("/api/dashboard/customers")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0
