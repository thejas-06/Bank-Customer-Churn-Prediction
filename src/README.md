# 📦 Core Machine Learning Engine (`src/`)

This package contains the core machine learning logic, modular data transformers, automated training scripts, and real-time inference services.

---

## 🏗️ Architecture & Component Flow

```
                     ┌──────────────────┐
                     │    config.py     │  ◄── Centralized Constants & Paths
                     └────────┬─────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
  ┌──────────────────┐                ┌──────────────────┐
  │   pipeline.py    │                │     train.py     │
  │ (Data Transform) │                │ (Model Training) │
  └─────────┬────────┘                └────────┬─────────┘
            │                                  │
            │          ┌───────────────┐       │
            └─────────►│  predict.py   │◄──────┘
                       │  (Inference)  │
                       └───────┬───────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   [ FastAPI Serving Layer ]            [ Streamlit Dashboard ]
```

---

## 📑 Module Responsibilities

### 1. `config.py` — Central Configuration
* **Purpose**: Single Source of Truth (SSOT) for the entire application.
* **Key Definitions**:
  * **Dynamic Path Resolution**: Cross-platform path references using Python's `pathlib.Path`.
  * **`FINAL_MODEL_FEATURES`**: The exact 12-column sequence expected by the XGBoost estimator to prevent column re-ordering bugs.
  * **`OPTIMAL_THRESHOLD = 0.655`**: The decision cutoff discovered via Precision-Recall curve analysis.
  * **`XGB_HYPERPARAMS`**: Hyperparameters tuned via 5-fold cross-validation (`scale_pos_weight=3.91`, `max_depth=5`, `learning_rate=0.01`, `n_estimators=300`).

---

### 2. `pipeline.py` — Feature Engineering Transformer
* **Purpose**: Scikit-Learn compatible transformer class `ChurnFeaturePipeline` (inherits from `BaseEstimator` and `TransformerMixin`).
* **Design Rationale**: Eliminates **Training-Serving Skew** by executing identical transformation logic during offline batch training and online single-record API requests.
* **Operations Executed**:
  * Strips identifier columns (`RowNumber`, `CustomerId`, `Surname`).
  * Computes `ZeroBalance`, `ProductActivity`, `BalanceSalaryRatio` (clipped at 99th percentile: `35.4762`), and `TenureAgeRatio` (`+0.5` additive smoothing).
  * Applies `OneHotEncoder(categories=[['France', 'Germany', 'Spain']], drop='first')` with explicit index alignment (`index=df.index`).
  * Drops non-predictive attributes (`HasCrCard`, `Tenure`, `EstimatedSalary`).

---

### 3. `train.py` — Automated Model Training & Serialization
* **Purpose**: Standalone, reproducible training pipeline executable via CLI or CI/CD jobs.
* **Execution Steps**:
  1. Ingests raw data from `data/Churn_Modelling.csv`.
  2. Fits `ChurnFeaturePipeline` and transforms features.
  3. Splits data (80/20) using **Stratification** (`stratify=y`) to maintain the 20.4% churn prior probability.
  4. Trains `XGBClassifier` with cost-sensitive loss weighting (`scale_pos_weight=3.91`).
  5. Evaluates model metrics at the calibrated `0.655` threshold.
  6. Samples a representative 50-row training background for SHAP.
  7. Serializes all compiled assets into `artifacts/` (`model.joblib`, `preprocessor.joblib`, `shap_explainer.joblib`, `threshold_config.json`).

---

### 4. `predict.py` — Real-Time Inference & Explainability Engine
* **Purpose**: High-performance prediction service with local TreeSHAP attribution and automated retention advice.
* **Key Patterns**:
  * **Singleton Pattern (`get_predictor()`)**: Loads binaries into RAM once on application boot and reuses the instance across all requests to eliminate disk I/O latency.
  * **`predict_single(customer_dict)`**: Accepts raw customer JSON, runs the pipeline, scores probability, checks against `0.655` cutoff, calculates top 3 SHAP drivers, and maps drivers to actionable banking retention playbooks.
  * **`predict_batch(df_raw)`**: High-throughput vectorized inference for bulk CSV spreadsheets.

---

## 🚀 How to Execute Training from CLI

```bash
python -m src.train
```

#### Expected Terminal Output:
```text
============================================================
>> Starting Production Model Training & Artifact Serialization
============================================================
Loading dataset from data/Churn_Modelling.csv...
Raw dataset shape: (10000, 14)
Fitting feature engineering pipeline...
Transformed feature matrix shape: (10000, 12)
Performing stratified train-test split (test_size=0.2, random_state=42)...
Training tuned XGBoost classifier...

--------------------------------------------------
Production Model Evaluation (Threshold = 0.655):
   * ROC-AUC Score   : 0.8704
   * Overall Accuracy: 0.8670 (86.7%)
   * Churn Precision : 0.7020 (70.2%)
   * Churn Recall    : 0.6020 (60.2%)
   * Churn F1-Score  : 0.6481
--------------------------------------------------
[SUCCESS] Saved Model: artifacts/model.joblib
[SUCCESS] Saved Preprocessor: artifacts/preprocessor.joblib
[SUCCESS] Saved SHAP Background: artifacts/shap_explainer.joblib
[SUCCESS] Saved Config: artifacts/threshold_config.json
```
