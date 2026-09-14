"""
Unit tests for the feature engineering and preprocessing pipeline.
"""
import pytest
import pandas as pd
import numpy as np

from src.pipeline import ChurnFeaturePipeline
from src.config import FINAL_MODEL_FEATURES, BALANCE_SALARY_CAP


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame([
        {
            "RowNumber": 1,
            "CustomerId": 15634602,
            "Surname": "Hargrave",
            "CreditScore": 619,
            "Geography": "France",
            "Gender": "Female",
            "Age": 42,
            "Tenure": 2,
            "Balance": 0.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 101348.88
        },
        {
            "RowNumber": 2,
            "CustomerId": 15647311,
            "Surname": "Hill",
            "CreditScore": 608,
            "Geography": "Spain",
            "Gender": "Female",
            "Age": 41,
            "Tenure": 0,  # Zero tenure edge case
            "Balance": 83807.86,
            "NumOfProducts": 2,
            "HasCrCard": 0,
            "IsActiveMember": 1,
            "EstimatedSalary": 10.0  # Extreme balance-to-salary outlier
        },
        {
            "RowNumber": 3,
            "CustomerId": 15619304,
            "Surname": "Onio",
            "CreditScore": 502,
            "Geography": "Germany",
            "Gender": "Male",
            "Age": 50,
            "Tenure": 8,
            "Balance": 159660.80,
            "NumOfProducts": 3,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 113931.57
        }
    ])


def test_pipeline_output_shape_and_columns(sample_raw_data):
    """Verifies that the pipeline produces the exact 12 expected columns in order."""
    pipeline = ChurnFeaturePipeline()
    pipeline.fit(sample_raw_data)
    transformed = pipeline.transform(sample_raw_data)

    assert transformed.shape == (3, 12)
    assert list(transformed.columns) == FINAL_MODEL_FEATURES


def test_zero_balance_feature(sample_raw_data):
    """Verifies that ZeroBalance is 1 when Balance is 0, and 0 otherwise."""
    pipeline = ChurnFeaturePipeline()
    pipeline.fit(sample_raw_data)
    transformed = pipeline.transform(sample_raw_data)

    # Row 0: Balance 0.0 -> ZeroBalance 1
    assert transformed.loc[0, "ZeroBalance"] == 1
    # Row 1: Balance 83807.86 -> ZeroBalance 0
    assert transformed.loc[1, "ZeroBalance"] == 0


def test_product_activity_interaction(sample_raw_data):
    """Verifies interaction: NumOfProducts * IsActiveMember."""
    pipeline = ChurnFeaturePipeline()
    pipeline.fit(sample_raw_data)
    transformed = pipeline.transform(sample_raw_data)

    # Row 0: 1 product * 1 active = 1
    assert transformed.loc[0, "ProductActivity"] == 1
    # Row 1: 2 products * 1 active = 2
    assert transformed.loc[1, "ProductActivity"] == 2
    # Row 2: 3 products * 0 inactive = 0
    assert transformed.loc[2, "ProductActivity"] == 0


def test_zero_tenure_smoothing(sample_raw_data):
    """Verifies additive +0.5 smoothing for zero tenure edge case."""
    pipeline = ChurnFeaturePipeline()
    pipeline.fit(sample_raw_data)
    transformed = pipeline.transform(sample_raw_data)

    # Row 1: Tenure 0, Age 41 -> (0 + 0.5) / 41 = 0.5 / 41
    expected_ratio = 0.5 / 41
    assert np.isclose(transformed.loc[1, "TenureAgeRatio"], expected_ratio)


def test_balance_salary_ratio_outlier_capping(sample_raw_data):
    """Verifies that extreme balance-to-salary ratios are clipped at 99th percentile cap."""
    pipeline = ChurnFeaturePipeline()
    pipeline.fit(sample_raw_data)
    transformed = pipeline.transform(sample_raw_data)

    # Row 1: 83807.86 / (10.0 + 1) = ~7618.9 -> should be capped at BALANCE_SALARY_CAP
    assert transformed.loc[1, "BalanceSalaryRatio"] <= BALANCE_SALARY_CAP
    assert transformed.loc[1, "BalanceSalaryRatio"] == BALANCE_SALARY_CAP


def test_categorical_encoding(sample_raw_data):
    """Verifies Gender label encoding and Geography one-hot encoding."""
    pipeline = ChurnFeaturePipeline()
    pipeline.fit(sample_raw_data)
    transformed = pipeline.transform(sample_raw_data)

    # Female: 0, Male: 1
    assert transformed.loc[0, "Gender_Encoded"] == 0
    assert transformed.loc[2, "Gender_Encoded"] == 1

    # Row 0: France -> Germany 0, Spain 0
    assert transformed.loc[0, "Geography_Germany"] == 0.0
    assert transformed.loc[0, "Geography_Spain"] == 0.0

    # Row 1: Spain -> Germany 0, Spain 1
    assert transformed.loc[1, "Geography_Germany"] == 0.0
    assert transformed.loc[1, "Geography_Spain"] == 1.0

    # Row 2: Germany -> Germany 1, Spain 0
    assert transformed.loc[2, "Geography_Germany"] == 1.0
    assert transformed.loc[2, "Geography_Spain"] == 0.0
