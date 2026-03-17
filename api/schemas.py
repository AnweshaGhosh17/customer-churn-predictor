"""
api/schemas.py
--------------
Pydantic models for request validation and response serialisation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ── Request Models ─────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """Input features for a single customer prediction."""

    # Core demographics
    SeniorCitizen:  int   = Field(0,     ge=0, le=1,     description="1 if senior citizen")
    Partner:        int   = Field(0,     ge=0, le=1,     description="1 if has partner")
    Dependents:     int   = Field(0,     ge=0, le=1,     description="1 if has dependents")

    # Account info
    tenure:         int   = Field(12,    ge=0,           description="Months as customer")
    Contract:       int   = Field(0,     ge=0, le=2,     description="0=M2M, 1=1yr, 2=2yr")
    PaperlessBilling: int = Field(1,     ge=0, le=1)
    PaymentMethod:  int   = Field(0,     ge=0, le=3,     description="0=Bank, 1=CreditCard, 2=ElectronicCheck, 3=MailedCheck")

    # Charges
    MonthlyCharges: float = Field(65.0,  ge=0)
    TotalCharges:   float = Field(780.0, ge=0)

    # Services
    PhoneService:   int   = Field(1,     ge=0, le=1)
    MultipleLines:  int   = Field(0,     ge=0, le=2,     description="0=No, 1=Yes, 2=NoPhone")
    InternetService: int  = Field(1,     ge=0, le=2,     description="0=DSL, 1=Fiber, 2=No")
    OnlineSecurity: int   = Field(0,     ge=0, le=2)
    OnlineBackup:   int   = Field(0,     ge=0, le=2)
    DeviceProtection: int = Field(0,     ge=0, le=2)
    TechSupport:    int   = Field(0,     ge=0, le=2)
    StreamingTV:    int   = Field(0,     ge=0, le=2)
    StreamingMovies: int  = Field(0,     ge=0, le=2)

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 8, "Contract": 0, "MonthlyCharges": 80.0,
                "TotalCharges": 640.0, "InternetService": 1,
                "TechSupport": 0, "PaymentMethod": 2
            }
        }


# ── Response Models ────────────────────────────────────────────────────────────

class SHAPFactor(BaseModel):
    feature:   str
    label:     str
    impact:    float
    direction: str


class Recommendation(BaseModel):
    action:   str
    reason:   str
    urgency:  str
    category: str


class PredictionResponse(BaseModel):
    probability:     float
    risk:            str
    clv_tier:        str
    priority:        str
    top_factors:     List[SHAPFactor]
    recommendations: List[Recommendation]
    segment:         Optional[dict] = None


class SimulateRequest(BaseModel):
    original:  CustomerFeatures
    modified:  CustomerFeatures


class SimulateResponse(BaseModel):
    original_probability:  float
    modified_probability:  float
    probability_delta:     float
    original_risk:         str
    modified_risk:         str
    risk_changed:          bool


class DashboardStats(BaseModel):
    total_customers:    int
    high_risk_count:    int
    medium_risk_count:  int
    low_risk_count:     int
    avg_probability:    float
    top_churn_drivers:  List[dict]
