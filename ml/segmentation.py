"""
ml/segmentation.py
------------------
Customer segmentation using K-Means clustering.
Run independently or call segment_customer() from the API.

FIXES vs original:
  - Absolute paths using __file__ so it works on Vercel.
  - Graceful fallback already existed; kept and improved.
"""

import os
import numpy as np
import pickle

_ML_DIR       = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(_ML_DIR, "model_artifacts")

SEGMENT_LABELS = {
    0: {"name": "Loyal High-Value",  "description": "Long tenure, high spend, low churn risk"},
    1: {"name": "Price Sensitive",   "description": "High charges relative to services used"},
    2: {"name": "At-Risk New Users", "description": "Short tenure, month-to-month contract"},
    3: {"name": "Low Engagement",    "description": "Few services, low interaction"},
}


def train_segmentation(processed_csv: str = None):
    """Train K-Means on processed data and save the model."""
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    if processed_csv is None:
        processed_csv = os.path.join(_ML_DIR, "..", "data", "processed", "churn_processed.csv")

    df = pd.read_csv(processed_csv)
    features = df[["tenure", "MonthlyCharges", "TotalCharges", "service_count"]].copy()

    seg_scaler = StandardScaler()
    X_scaled   = seg_scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    pickle.dump(kmeans,     open(os.path.join(ARTIFACTS_DIR, "kmeans.pkl"),     "wb"))
    pickle.dump(seg_scaler, open(os.path.join(ARTIFACTS_DIR, "seg_scaler.pkl"), "wb"))
    print("Segmentation model saved.")
    return kmeans, seg_scaler


def segment_customer(features: dict) -> dict:
    """Assign a single customer to a segment. Returns Unknown if model not trained."""
    kmeans_path     = os.path.join(ARTIFACTS_DIR, "kmeans.pkl")
    seg_scaler_path = os.path.join(ARTIFACTS_DIR, "seg_scaler.pkl")

    if not os.path.exists(kmeans_path) or not os.path.exists(seg_scaler_path):
        return {
            "segment_id":   -1,
            "segment_name": "Unknown",
            "description":  "Segmentation model not trained yet — run ml/segmentation.py first",
        }

    try:
        kmeans     = pickle.load(open(kmeans_path,     "rb"))
        seg_scaler = pickle.load(open(seg_scaler_path, "rb"))
    except Exception as e:
        return {"segment_id": -1, "segment_name": "Unknown", "description": str(e)}

    X = np.array([[
        features.get("tenure", 0),
        features.get("MonthlyCharges", 0),
        features.get("TotalCharges", 0),
        features.get("service_count", 0),
    ]])
    X_scaled   = seg_scaler.transform(X)
    segment_id = int(kmeans.predict(X_scaled)[0])
    label      = SEGMENT_LABELS.get(segment_id, {"name": "Unknown", "description": ""})

    return {
        "segment_id":   segment_id,
        "segment_name": label["name"],
        "description":  label["description"],
    }


if __name__ == "__main__":
    train_segmentation()
