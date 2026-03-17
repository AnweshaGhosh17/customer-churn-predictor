"""
api/routes/dashboard.py
-----------------------
GET /api/dashboard  — Aggregate stats for the risk dashboard UI.
Uses sample data when no live database is connected (hackathon mode).
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import random
from fastapi import APIRouter
from api.schemas import DashboardStats

router = APIRouter()

# Sample pre-computed stats (replace with real DB queries in production)
SAMPLE_CUSTOMERS = [
    {"id": f"C{1000+i}", "tenure": random.randint(1, 72),
     "monthly_charges": round(random.uniform(20, 110), 2),
     "contract": random.choice([0, 1, 2]),
     "probability": round(random.uniform(0.05, 0.95), 2)}
    for i in range(50)
]

TOP_DRIVERS = [
    {"feature": "Contract type",     "importance": 0.31},
    {"feature": "Monthly charges",   "importance": 0.22},
    {"feature": "Tenure",            "importance": 0.18},
    {"feature": "Tech support",      "importance": 0.12},
    {"feature": "Internet service",  "importance": 0.10},
    {"feature": "Payment method",    "importance": 0.07},
]


@router.get("/dashboard", response_model=DashboardStats)
async def dashboard():
    """Return aggregate churn stats for the dashboard page."""
    high   = [c for c in SAMPLE_CUSTOMERS if c["probability"] >= 0.65]
    medium = [c for c in SAMPLE_CUSTOMERS if 0.35 <= c["probability"] < 0.65]
    low    = [c for c in SAMPLE_CUSTOMERS if c["probability"] < 0.35]
    avg_p  = round(sum(c["probability"] for c in SAMPLE_CUSTOMERS) / len(SAMPLE_CUSTOMERS), 4)

    return DashboardStats(
        total_customers=len(SAMPLE_CUSTOMERS),
        high_risk_count=len(high),
        medium_risk_count=len(medium),
        low_risk_count=len(low),
        avg_probability=avg_p,
        top_churn_drivers=TOP_DRIVERS,
    )


@router.get("/dashboard/customers")
async def get_customers():
    """Return the sample customer list with risk labels."""
    def label(p):
        return "HIGH" if p >= 0.65 else "MEDIUM" if p >= 0.35 else "LOW"

    return [
        {**c, "risk": label(c["probability"])}
        for c in sorted(SAMPLE_CUSTOMERS, key=lambda x: -x["probability"])
    ]
