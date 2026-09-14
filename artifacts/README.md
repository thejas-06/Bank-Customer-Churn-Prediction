# 💾 Model Registry & Serialized Artifacts (`artifacts/`)

This directory stores the frozen, compiled binary assets produced by `src/train.py` and consumed by the real-time serving layers.

---

## 📑 Serialized Artifacts Inventory

| File | Type | Size | Description |
|---|---|---|---|
| **`model.joblib`** | Binary (Joblib) | ~728 KB | Tuned `XGBClassifier` estimator with 300 gradient-boosted decision trees and learned leaf weights. |
| **`preprocessor.joblib`** | Binary (Joblib) | ~1.5 KB | Fitted `ChurnFeaturePipeline` containing the `OneHotEncoder` categories and transformation rules. |
| **`shap_explainer.joblib`** | Binary (Joblib) | ~6.6 KB | 50-sample representative background distribution used by TreeSHAP for local feature attributions. |
| **`threshold_config.json`** | JSON Metadata | 220 Bytes | Calibrated `0.655` decision threshold and evaluation metrics record. |

---

## 🔒 Metadata Schema (`threshold_config.json`)

```json
{
  "model_name": "Tuned XGBoost Classifier",
  "optimal_threshold": 0.655,
  "metrics": {
    "roc_auc": 0.8704,
    "accuracy": 0.867,
    "precision": 0.702,
    "recall": 0.602,
    "f1_score": 0.6481
  }
}
```

---

## 🔄 How to Recompile Artifacts

Artifacts are compiled automatically during offline training or Docker build:
```bash
python -m src.train
```
