"""
Configuration parameters, file paths, and model hyperparameters for Churn Prediction.
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Data file paths
RAW_DATA_PATH = DATA_DIR / "Churn_Modelling.csv"
CLEANED_DATA_PATH = DATA_DIR / "churn_cleaned.csv"

# Artifact file paths
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.joblib"
CONFIG_PATH = ARTIFACTS_DIR / "threshold_config.json"

# Feature definitions
IDENTIFIER_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
TARGET_COLUMN = "Exited"

# Raw features expected from customer input
RAW_NUMERICAL_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]
RAW_CATEGORICAL_FEATURES = ["Geography", "Gender"]

# Exact 12 final features expected by the trained model (in exact column order)
FINAL_MODEL_FEATURES = [
    "CreditScore",
    "Age",
    "Balance",
    "NumOfProducts",
    "IsActiveMember",
    "ZeroBalance",
    "ProductActivity",
    "BalanceSalaryRatio",
    "Gender_Encoded",
    "Geography_Germany",
    "Geography_Spain",
    "TenureAgeRatio"
]

# Business-calibrated optimal decision threshold (maximizes F1 while delivering 70% Precision & 60% Recall)
OPTIMAL_THRESHOLD = 0.655

# Outlier capping threshold for BalanceSalaryRatio (99th percentile)
BALANCE_SALARY_CAP = 35.4762

# Optimal tuned XGBoost hyperparameters
XGB_HYPERPARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 1,
    "scale_pos_weight": 3.91,
    "random_state": 42,
    "eval_metric": "logloss"
}
