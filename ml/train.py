"""
ml/train.py
-----------
Full training pipeline for customer churn prediction.
Run this script (or the Colab notebook) to generate model artifacts.

Usage:
    python ml/train.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DATA_PATH   = "data/raw/telco_churn.csv"
PROC_DATA_PATH  = "data/processed/churn_processed.csv"
ARTIFACTS_DIR   = "ml/model_artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(RAW_DATA_PATH)
print(f"  Shape: {df.shape}")

# ── 2. Clean ───────────────────────────────────────────────────────────────────
print("Cleaning...")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.drop(columns=["customerID"], inplace=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# ── 3. Feature Engineering ────────────────────────────────────────────────────
print("Engineering features...")

# Charges per month ratio (catches overpaying customers)
df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

# Service count (more services = more sticky)
service_cols = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]
df["service_count"] = df[service_cols].apply(
    lambda row: sum(1 for v in row if v not in ["No", "No internet service", "No phone service"]),
    axis=1
)

# Long-term customer flag
df["is_long_term"] = (df["tenure"] > 24).astype(int)

# ── 4. Encode Categoricals ────────────────────────────────────────────────────
print("Encoding categorical columns...")
le_dict = {}
cat_cols = df.select_dtypes(include="object").columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# ── 5. Scale Numerics ─────────────────────────────────────────────────────────
num_cols = ["tenure", "MonthlyCharges", "TotalCharges",
            "charges_per_tenure", "service_count"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save processed dataset
df.to_csv(PROC_DATA_PATH, index=False)
print(f"  Saved processed data → {PROC_DATA_PATH}")

# ── 6. Train / Test Split ─────────────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

# ── 7. Train XGBoost ──────────────────────────────────────────────────────────
print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── 8. Evaluate ───────────────────────────────────────────────────────────────
print("\n── Evaluation ─────────────────────────────────────────────")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"  5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 9. Save Artifacts ─────────────────────────────────────────────────────────
print("\nSaving artifacts...")
pickle.dump(model,    open(f"{ARTIFACTS_DIR}/xgb_model.pkl", "wb"))
pickle.dump(scaler,   open(f"{ARTIFACTS_DIR}/scaler.pkl",    "wb"))
pickle.dump(le_dict,  open(f"{ARTIFACTS_DIR}/label_encoders.pkl", "wb"))
pickle.dump(list(X.columns), open(f"{ARTIFACTS_DIR}/feature_names.pkl", "wb"))

print(f"  ✓ xgb_model.pkl")
print(f"  ✓ scaler.pkl")
print(f"  ✓ label_encoders.pkl")
print(f"  ✓ feature_names.pkl")
print("\nDone! All artifacts saved to ml/model_artifacts/")
