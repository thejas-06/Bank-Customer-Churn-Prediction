"""
Pydantic schemas for data validation and API request/response structures.
"""
from typing import List, Literal
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850, description="Customer credit rating (300 - 850)", json_schema_extra={"example": 619})
    Geography: Literal["France", "Germany", "Spain"] = Field(..., description="Country of residence", json_schema_extra={"example": "France"})
    Gender: Literal["Female", "Male"] = Field(..., description="Customer biological gender", json_schema_extra={"example": "Female"})
    Age: int = Field(..., ge=18, le=100, description="Customer age in years (18 - 100)", json_schema_extra={"example": 42})
    Tenure: int = Field(..., ge=0, le=10, description="Years customer has held an account (0 - 10)", json_schema_extra={"example": 2})
    Balance: float = Field(..., ge=0.0, description="Account liquid balance amount", json_schema_extra={"example": 0.0})
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of bank products subscribed to (1 - 4)", json_schema_extra={"example": 1})
    HasCrCard: int = Field(..., ge=0, le=1, description="Whether customer holds a credit card (1=Yes, 0=No)", json_schema_extra={"example": 1})
    IsActiveMember: int = Field(..., ge=0, le=1, description="Whether customer is an active member (1=Yes, 0=No)", json_schema_extra={"example": 1})
    EstimatedSalary: float = Field(..., ge=0.0, description="Estimated customer annual earnings", json_schema_extra={"example": 101348.88})


class FeatureImpact(BaseModel):
    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value for this customer")
    shap_impact: float = Field(..., description="SHAP attribution value (positive increases churn probability)")
    direction: str = Field(..., description="'Increases Risk' or 'Decreases Risk'")


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Estimated churn probability between 0.0 and 1.0")
    will_churn: bool = Field(..., description="Binary classification based on calibrated decision threshold")
    decision_threshold: float = Field(..., description="Decision cutoff threshold applied (e.g. 0.655)")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Categorical risk rating")
    top_risk_factors: List[FeatureImpact] = Field(..., description="Top drivers pushing customer toward churn")
    retention_recommendation: str = Field(..., description="Automated business action recommendation")


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerInput] = Field(..., min_length=1, description="List of customer records to evaluate")


class BatchCustomerResult(BaseModel):
    customer_index: int
    churn_probability: float
    will_churn: bool
    risk_level: str


class BatchPredictionResponse(BaseModel):
    total_customers: int
    predicted_churn_count: int
    churn_rate_percent: float
    results: List[BatchCustomerResult]
