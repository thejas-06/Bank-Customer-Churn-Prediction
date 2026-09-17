# ⚡ FastAPI Model Serving Microservice (`api/`)

This directory contains the production REST API for real-time and batch customer churn inference, schema validation, and health monitoring.

---

## 📑 Module Overview

| File | Responsibility |
|---|---|
| **`schemas.py`** | Pydantic v2 data models enforcing strict types, range boundaries, and JSON contracts. |
| **`main.py`** | FastAPI application routing, CORS configuration, error handling, and endpoint handlers. |

---

## 🌐 Endpoint Reference

### 1. `GET /health` — Service Liveness & Readiness Probe
* **Purpose**: Used by Kubernetes, Docker healthchecks, and AWS load balancers to monitor service health.
* **Response `200 OK`**:
```json
{
  "status": "healthy",
  "service": "Bank Customer Churn Prediction API",
  "model_ready": true,
  "decision_threshold": 0.655
}
```

---

### 2. `POST /predict` — Real-Time Single Customer Inference
* **Purpose**: Evaluates an individual customer profile, returns calibrated churn risk, local TreeSHAP drivers, and retention advice.
* **Latency**: `< 15ms`
* **Request Payload**:
```json
{
  "CreditScore": 650,
  "Geography": "Germany",
  "Gender": "Female",
  "Age": 52,
  "Tenure": 3,
  "Balance": 120000.0,
  "NumOfProducts": 3,
  "HasCrCard": 1,
  "IsActiveMember": 0,
  "EstimatedSalary": 95000.0
}
```

* **Response `200 OK`**:
```json
{
  "churn_probability": 0.8142,
  "will_churn": true,
  "decision_threshold": 0.655,
  "risk_level": "High",
  "top_risk_factors": [
    {
      "feature": "NumOfProducts",
      "value": 3.0,
      "shap_impact": 0.1824,
      "direction": "Increases Risk"
    },
    {
      "feature": "Age",
      "value": 52.0,
      "shap_impact": 0.1412,
      "direction": "Increases Risk"
    }
  ],
  "retention_recommendation": "High multi-product complexity detected: Simplify account bundle or assign dedicated account specialist. | Dormant account status: Send customized mobile app re-engagement campaign with fee waiver incentives."
}
```

---

### 3. `POST /predict/batch` — High-Throughput Batch Scoring
* **Purpose**: Processes lists of customer records and returns individual predictions alongside aggregate cohort statistics.
* **Request Payload**:
```json
{
  "customers": [
    {
      "CreditScore": 600,
      "Geography": "France",
      "Gender": "Female",
      "Age": 40,
      "Tenure": 2,
      "Balance": 0.0,
      "NumOfProducts": 1,
      "HasCrCard": 1,
      "IsActiveMember": 1,
      "EstimatedSalary": 70000.0
    }
  ]
}
```

* **Response `200 OK`**:
```json
{
  "total_customers": 1,
  "predicted_churn_count": 0,
  "churn_rate_percent": 0.0,
  "results": [
    {
      "customer_index": 0,
      "churn_probability": 0.1245,
      "will_churn": false,
      "risk_level": "Low"
    }
  ]
}
```

---

## 🔒 Input Validation & Error Handling (Pydantic v2)

If a client sends out-of-range or invalid data, the request is intercepted before hitting the ML model and returns **`422 Unprocessable Entity`**:

* `CreditScore < 300` or `> 850` $\rightarrow$ Rejected.
* `Age < 18` or `> 100` $\rightarrow$ Rejected.
* `Geography` not in `["France", "Germany", "Spain"]` $\rightarrow$ Rejected.

---

## 🚀 How to Run Locally

```bash
# Start Uvicorn ASGI Server
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

* **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Alternative Redoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
