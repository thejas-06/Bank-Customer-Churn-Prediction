"""
Feature engineering and transformation pipeline for Bank Customer Churn data.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from src.config import (
    IDENTIFIER_COLUMNS,
    FINAL_MODEL_FEATURES,
    BALANCE_SALARY_CAP
)


class ChurnFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies all domain feature engineering,
    outlier clipping, and categorical encodings to raw banking customer data.
    """

    def __init__(self, balance_salary_cap: float = BALANCE_SALARY_CAP):
        self.balance_salary_cap = balance_salary_cap
        self.geo_encoder = OneHotEncoder(
            categories=[['France', 'Germany', 'Spain']],
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
        )
        self.gender_map = {'Female': 0, 'Male': 1}
        self.is_fitted = False

    def fit(self, X, y=None):
        """Fit internal encoders on raw input data."""
        df = X.copy()
        if isinstance(df, dict):
            df = pd.DataFrame([df])

        if 'Geography' in df.columns:
            self.geo_encoder.fit(df[['Geography']])
        
        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Transforms raw customer DataFrame/dict into the exact 12-feature matrix
        expected by the trained XGBoost model.
        """
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        # 1. Drop administrative identifiers if present
        cols_to_drop = [col for col in IDENTIFIER_COLUMNS if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # 2. Domain Feature Engineering
        # Feature 1: Zero Balance Indicator
        df['ZeroBalance'] = (df['Balance'] == 0).astype(int)

        # Feature 2: Product & Member Activity Interaction
        df['ProductActivity'] = df['NumOfProducts'] * df['IsActiveMember']

        # Feature 3: Balance-to-Salary Ratio (with +1 division-by-zero defense & 99th percentile cap)
        raw_ratio = df['Balance'] / (df['EstimatedSalary'] + 1.0)
        df['BalanceSalaryRatio'] = raw_ratio.clip(upper=self.balance_salary_cap)

        # Feature 4: Tenure-to-Age Relative Loyalty Ratio (+0.5 smoothing for 0 tenure)
        df['TenureAgeRatio'] = (df['Tenure'] + 0.5) / df['Age']

        # 3. Categorical Encodings
        # Binary Gender mapping (Female: 0, Male: 1)
        if 'Gender' in df.columns:
            df['Gender_Encoded'] = df['Gender'].map(self.gender_map).fillna(0).astype(int)

        # Geography One-Hot Encoding (France is dropped base category; Germany & Spain output)
        if 'Geography' in df.columns:
            geo_features = self.geo_encoder.transform(df[['Geography']])
            geo_df = pd.DataFrame(
                geo_features,
                columns=['Geography_Germany', 'Geography_Spain'],
                index=df.index
            )
            df = pd.concat([df, geo_df], axis=1)

        # 4. Drop non-predictive & redundant raw features
        prune_cols = ['Geography', 'Gender', 'HasCrCard', 'Tenure', 'EstimatedSalary', 'AgeGroup']
        drop_existing = [col for col in prune_cols if col in df.columns]
        if drop_existing:
            df = df.drop(columns=drop_existing)

        # 5. Guarantee strict column order and completeness
        for col in FINAL_MODEL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        return df[FINAL_MODEL_FEATURES]
