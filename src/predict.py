"""
Inference and explainability engine for Bank Customer Churn Prediction.
"""
import json
import joblib
import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, List, Union

from src.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    EXPLAINER_PATH,
    CONFIG_PATH,
    OPTIMAL_THRESHOLD,
    FINAL_MODEL_FEATURES
)


class ChurnPredictor:
    """
    Production-ready inference service that transforms customer data,
    predicts churn probabilities, applies calibrated decision boundaries,
    and extracts local SHAP explainability insights.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.background_sample = None
        self.explainer = None
        self.threshold = OPTIMAL_THRESHOLD
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model, preprocessor, and SHAP background from artifacts/."""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            self.background_sample = joblib.load(EXPLAINER_PATH)
            
            # Instantiate KernelExplainer with clean lambda wrapper
            self.explainer = shap.KernelExplainer(
                lambda x: self.model.predict_proba(x),
                self.background_sample
            )
            
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, "r") as f:
                    config = json.load(f)
                    self.threshold = config.get("optimal_threshold", OPTIMAL_THRESHOLD)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model artifacts from artifacts/. "
                f"Please ensure you run 'python -m src.train' first. Error: {e}"
            )

    def _get_risk_level(self, probability: float) -> str:
        if probability >= 0.70:
            return "High"
        elif probability >= 0.40:
            return "Medium"
        return "Low"

    def _get_retention_advice(self, customer_raw: Dict[str, Any], top_factors: List[Dict[str, Any]]) -> str:
        factors = [f['feature'] for f in top_factors]
        num_products = customer_raw.get('NumOfProducts', 1)
        is_active = customer_raw.get('IsActiveMember', 1)
        balance = customer_raw.get('Balance', 0.0)
        age = customer_raw.get('Age', 40)

        recommendations = []

        if num_products >= 3:
            recommendations.append("High multi-product complexity detected: Simplify account bundle or assign dedicated account specialist.")
        if is_active == 0:
            recommendations.append("Dormant account status: Send customized mobile app re-engagement campaign with fee waiver incentives.")
        if age >= 50:
            recommendations.append("Senior customer segment: Provide premier customer service support and tailored wealth management consultation.")
        if balance == 0:
            recommendations.append("Zero balance account: Offer introductory high-yield savings deposit promotion.")

        if not recommendations:
            recommendations.append("Standard loyalty retention: Maintain regular engagement and review quarterly satisfaction surveys.")

        return " | ".join(recommendations)

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs end-to-end inference and SHAP explainability for a single customer payload.
        """
        # 1. Transform raw customer dict into 12-feature matrix
        input_df = pd.DataFrame([customer_data])
        X_trans = self.preprocessor.transform(input_df)

        # 2. Model Prediction
        prob = float(self.model.predict_proba(X_trans)[0, 1])
        will_churn = bool(prob >= self.threshold)
        risk_level = self._get_risk_level(prob)

        # 3. Local SHAP Attribution (Extract class 1: Churn contribution)
        shap_vals = self.explainer.shap_values(X_trans)
        if isinstance(shap_vals, list):
            sample_shap = shap_vals[1][0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            sample_shap = shap_vals[0, :, 1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            sample_shap = shap_vals[0]
        else:
            sample_shap = np.zeros(len(FINAL_MODEL_FEATURES))

        feature_contributions = []
        for feat_name, shap_val, feat_val in zip(FINAL_MODEL_FEATURES, sample_shap, X_trans.iloc[0]):
            feature_contributions.append({
                "feature": feat_name,
                "value": round(float(feat_val), 4),
                "shap_impact": round(float(shap_val), 4),
                "direction": "Increases Risk" if shap_val > 0 else "Decreases Risk"
            })

        # Sort by absolute SHAP impact
        feature_contributions.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
        top_risk_drivers = [f for f in feature_contributions if f["shap_impact"] > 0][:3]

        recommendation = self._get_retention_advice(customer_data, top_risk_drivers)

        return {
            "churn_probability": round(prob, 4),
            "will_churn": will_churn,
            "decision_threshold": self.threshold,
            "risk_level": risk_level,
            "top_risk_factors": top_risk_drivers,
            "all_feature_impacts": feature_contributions,
            "retention_recommendation": recommendation
        }

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Runs high-throughput batch scoring on a DataFrame of raw customer records.
        """
        X_trans = self.preprocessor.transform(df_raw)
        probs = self.model.predict_proba(X_trans)[:, 1]
        
        results_df = df_raw.copy()
        results_df["Churn_Probability"] = np.round(probs, 4)
        results_df["Predicted_Churn"] = (probs >= self.threshold).astype(int)
        results_df["Risk_Level"] = [self._get_risk_level(p) for p in probs]
        
        return results_df


# Singleton instance for fast caching in API & Web Apps
_predictor_instance = None


def get_predictor() -> ChurnPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ChurnPredictor()
    return _predictor_instance
