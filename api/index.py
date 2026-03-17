"""
api/index.py
------------
Main FastAPI application entrypoint.
Vercel calls this file directly as a serverless function.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predict, simulate, dashboard

app = FastAPI(
    title="Customer Churn Predictor API",
    description="Predict customer churn probability with SHAP explanations and retention recommendations.",
    version="1.0.0",
)

# Allow frontend to call the API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(predict.router,   prefix="/api", tags=["Prediction"])
app.include_router(simulate.router,  prefix="/api", tags=["Simulation"])
app.include_router(dashboard.router, prefix="/api", tags=["Dashboard"])


@app.get("/")
async def root():
    return {
        "message": "Customer Churn Predictor API",
        "docs":    "/docs",
        "endpoints": ["/api/predict", "/api/simulate", "/api/dashboard"]
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
