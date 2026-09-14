"""
Integration tests for FastAPI endpoints and input validation schemas.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test service health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_ready"] is True
    assert "decision_threshold" in data


def test_predict_single_valid_customer():
    """Test prediction for a valid customer payload."""
    payload = {
        "CreditScore": 650,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 45,
        "Tenure": 3,
        "Balance": 120000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 95000.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["will_churn"], bool)
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert "top_risk_factors" in data
    assert "retention_recommendation" in data


def test_predict_invalid_credit_score():
    """Test validation failure when CreditScore is below minimum (300)."""
    payload = {
        "CreditScore": 250,  # Invalid: below 300
        "Geography": "France",
        "Gender": "Male",
        "Age": 30,
        "Tenure": 2,
        "Balance": 50000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 60000.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_invalid_geography():
    """Test validation failure when Geography is not in the allowed list."""
    payload = {
        "CreditScore": 700,
        "Geography": "Canada",  # Invalid country
        "Gender": "Male",
        "Age": 35,
        "Tenure": 4,
        "Balance": 30000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 80000.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch():
    """Test batch prediction endpoint."""
    batch_payload = {
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
            },
            {
                "CreditScore": 750,
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 55,
                "Tenure": 6,
                "Balance": 140000.0,
                "NumOfProducts": 3,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 120000.0
            }
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_customers"] == 2
    assert "predicted_churn_count" in data
    assert "churn_rate_percent" in data
    assert len(data["results"]) == 2
