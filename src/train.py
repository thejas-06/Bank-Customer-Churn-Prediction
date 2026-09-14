"""
Training and serialization script for the Bank Customer Churn prediction model.
"""
import json
import os
import joblib
import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from src.config import (
    RAW_DATA_PATH,
    ARTIFACTS_DIR,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    EXPLAINER_PATH,
    CONFIG_PATH,
    TARGET_COLUMN,
    OPTIMAL_THRESHOLD,
    XGB_HYPERPARAMS
)
from src.pipeline import ChurnFeaturePipeline


def train_and_export():
    print("=" * 60)
    print(">> Starting Production Model Training & Artifact Serialization")
    print("=" * 60)

    # 1. Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 2. Load Raw Data
    print(f"Loading dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw dataset shape: {df.shape}")

    # 3. Separate features and target
    X_raw = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 4. Fit Feature Pipeline & Transform
    print("Fitting feature engineering pipeline...")
    pipeline = ChurnFeaturePipeline()
    pipeline.fit(X_raw)
    X_transformed = pipeline.transform(X_raw)
    print(f"Transformed feature matrix shape: {X_transformed.shape}")

    # 5. Stratified Train-Test Split (80/20)
    print("Performing stratified train-test split (test_size=0.2, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, stratify=y, random_state=42
    )

    # 6. Train Tuned XGBoost Model
    print("Training tuned XGBoost classifier...")
    xgb_model = XGBClassifier(**XGB_HYPERPARAMS)
    xgb_model.fit(X_train, y_train)

    # 7. Model Evaluation
    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    y_pred_calibrated = (y_proba >= OPTIMAL_THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred_calibrated)
    prec = precision_score(y_test, y_pred_calibrated, pos_label=1)
    rec = recall_score(y_test, y_pred_calibrated, pos_label=1)
    f1 = f1_score(y_test, y_pred_calibrated, pos_label=1)

    print("\n" + "-" * 50)
    print(f"Production Model Evaluation (Threshold = {OPTIMAL_THRESHOLD:.3f}):")
    print(f"   * ROC-AUC Score   : {roc_auc:.4f}")
    print(f"   * Overall Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"   * Churn Precision : {prec:.4f} ({prec*100:.1f}%)")
    print(f"   * Churn Recall    : {rec:.4f} ({rec*100:.1f}%)")
    print(f"   * Churn F1-Score  : {f1:.4f}")
    print("-" * 50)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_calibrated))

    # 8. Sample background distribution for SHAP explainability
    print("Extracting representative background sample for SHAP explainability...")
    background_sample = shap.sample(X_train, 50, random_state=42)

    # 9. Save Artifacts
    print("Saving production artifacts to artifacts/ ...")
    joblib.dump(xgb_model, MODEL_PATH)
    print(f"   [SUCCESS] Saved Model: {MODEL_PATH}")

    joblib.dump(pipeline, PREPROCESSOR_PATH)
    print(f"   [SUCCESS] Saved Preprocessor: {PREPROCESSOR_PATH}")

    joblib.dump(background_sample, EXPLAINER_PATH)
    print(f"   [SUCCESS] Saved SHAP Background Distribution: {EXPLAINER_PATH}")

    threshold_config = {
        "model_name": "Tuned XGBoost Classifier",
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "metrics": {
            "roc_auc": round(float(roc_auc), 4),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4)
        }
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(threshold_config, f, indent=2)
    print(f"   [SUCCESS] Saved Config: {CONFIG_PATH}")

    print("\nTraining and artifact generation completed successfully!")


if __name__ == "__main__":
    train_and_export()
