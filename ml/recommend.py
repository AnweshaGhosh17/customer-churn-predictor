"""
ml/recommend.py
---------------
Rule-based retention recommendation engine.
Uses model predictions + customer features to suggest actions.
"""

from typing import List, Dict


def get_recommendations(features: dict, probability: float, risk: str, clv_tier: str) -> List[Dict]:
    """
    Returns a ranked list of recommended retention actions.

    Args:
        features:    Raw customer features dict
        probability: Churn probability (0–1)
        risk:        "HIGH" | "MEDIUM" | "LOW"
        clv_tier:    "HIGH" | "MEDIUM" | "LOW"

    Returns:
        List of recommendation dicts with action, reason, urgency
    """
    recommendations = []

    # ── Contract upgrade ──────────────────────────────────────────────────────
    # Contract: 0 = Month-to-month, 1 = One year, 2 = Two year
    contract = features.get("Contract", 0)
    if contract == 0:
        discount = "20%" if clv_tier == "HIGH" else "15%" if clv_tier == "MEDIUM" else "10%"
        recommendations.append({
            "action":   f"Offer {discount} discount on annual contract upgrade",
            "reason":   "Month-to-month customers churn 3x more than annual",
            "urgency":  "HIGH",
            "category": "Contract",
        })

    # ── High monthly charges ──────────────────────────────────────────────────
    monthly = features.get("MonthlyCharges", 0)
    if monthly > 75:
        recommendations.append({
            "action":   "Suggest a lower-cost service bundle",
            "reason":   f"Monthly charges of ${monthly:.0f} are above average",
            "urgency":  "HIGH" if monthly > 90 else "MEDIUM",
            "category": "Pricing",
        })

    # ── New customer (short tenure) ───────────────────────────────────────────
    tenure = features.get("tenure", 12)
    if tenure < 12:
        recommendations.append({
            "action":   "Assign dedicated onboarding support for first 90 days",
            "reason":   "Customers in their first year churn at the highest rate",
            "urgency":  "HIGH",
            "category": "Onboarding",
        })

    # ── No tech support ───────────────────────────────────────────────────────
    tech_support = features.get("TechSupport", 0)
    if tech_support == 0 and features.get("InternetService", 0) != 0:
        recommendations.append({
            "action":   "Offer 3-month free trial of Tech Support add-on",
            "reason":   "Customers without support report higher frustration",
            "urgency":  "MEDIUM",
            "category": "Product",
        })

    # ── No online security ────────────────────────────────────────────────────
    security = features.get("OnlineSecurity", 0)
    if security == 0 and features.get("InternetService", 0) != 0:
        recommendations.append({
            "action":   "Highlight Online Security features — offer 1 month free",
            "reason":   "Security add-ons increase perceived value significantly",
            "urgency":  "LOW",
            "category": "Product",
        })

    # ── High-value loyal customer ─────────────────────────────────────────────
    if clv_tier == "HIGH" and tenure > 24:
        recommendations.append({
            "action":   "Enroll in VIP loyalty program with priority support",
            "reason":   "High-value long-term customers respond well to recognition",
            "urgency":  "HIGH",
            "category": "Loyalty",
        })

    # ── Electronic check (highest churn payment method) ───────────────────────
    payment = features.get("PaymentMethod", 0)
    if payment == 2:  # Electronic check
        recommendations.append({
            "action":   "Nudge toward auto-pay with a $5/month discount",
            "reason":   "Electronic check users have the highest churn rate",
            "urgency":  "MEDIUM",
            "category": "Payment",
        })

    # ── Default fallback ──────────────────────────────────────────────────────
    if not recommendations:
        recommendations.append({
            "action":   "Send personalised loyalty thank-you offer",
            "reason":   "Proactive engagement reduces passive churn",
            "urgency":  "LOW",
            "category": "Loyalty",
        })

    # Sort by urgency: HIGH → MEDIUM → LOW
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recommendations.sort(key=lambda r: order.get(r["urgency"], 3))

    return recommendations[:3]  # Return top 3 actions
