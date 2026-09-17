"""
FastAPI application for serving Bank Customer Churn predictions.
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from api.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchCustomerResult
)
from src.predict import get_predictor

app = FastAPI(
    title="🏦 Bank Customer Churn Prediction Microservice",
    description=(
        "Production REST API for real-time customer attrition prediction, "
        "calibrated threshold classification, and TreeSHAP explainability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    tags=["Root"],
    summary="Root Welcome Endpoint"
)
def root():
    """Welcome endpoint with links to interactive documentation and health check."""
    return {
        "message": "Welcome to the Bank Customer Churn Prediction API",
        "documentation": "/docs",
        "health_check": "/health",
        "version": "1.0.0"
    }


@app.get(
    "/health",
    tags=["System Health"],
    summary="Service Liveness and Model Readiness Check"
)
def health_check():
    """Confirms API liveness and verifies that the model artifacts are loaded."""
    try:
        predictor = get_predictor()
        return {
            "status": "healthy",
            "service": "Bank Customer Churn Prediction API",
            "model_ready": predictor.model is not None,
            "decision_threshold": predictor.threshold
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model engine initialization failed: {str(e)}"
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Predict churn probability and SHAP attribution for a single customer"
)
def predict_single_customer(customer: CustomerInput):
    """
    Accepts raw customer demographic and financial attributes, runs full feature
    engineering, and returns churn risk score with local SHAP explainability.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict_single(customer.model_dump())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="High-throughput batch churn prediction for multiple customers"
)
def predict_batch_customers(batch_request: BatchPredictionRequest):
    """
    Processes a list of customer records in a vectorized batch and returns
    individual predictions alongside summary aggregate churn statistics.
    """
    try:
        predictor = get_predictor()
        raw_records = [c.model_dump() for c in batch_request.customers]
        df_raw = pd.DataFrame(raw_records)
        
        scored_df = predictor.predict_batch(df_raw)

        results = []
        churn_count = 0
        for idx, row in scored_df.iterrows():
            is_churn = bool(row["Predicted_Churn"] == 1)
            if is_churn:
                churn_count += 1
            results.append(
                BatchCustomerResult(
                    customer_index=int(idx),
                    churn_probability=float(row["Churn_Probability"]),
                    will_churn=is_churn,
                    risk_level=str(row["Risk_Level"])
                )
            )

        total = len(results)
        churn_rate = round((churn_count / total) * 100, 2) if total > 0 else 0.0

        return BatchPredictionResponse(
            total_customers=total,
            predicted_churn_count=churn_count,
            churn_rate_percent=churn_rate,
            results=results
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )
