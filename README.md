# 🏦 Bank Customer Churn Prediction & Explainability Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Pytest](https://img.shields.io/badge/Tests-Pytest-green.svg?logo=pytest&logoColor=white)](https://pytest.org/)

An end-to-end, production-grade Machine Learning system designed to identify retail banking customers at risk of attrition, quantify financial exposure, explain individual prediction drivers via **TreeSHAP**, and deliver actionable retention strategies through a **FastAPI microservice** and an **interactive Streamlit stakeholder dashboard**.

---

## 🏗️ System Architecture

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │           END-TO-END SYSTEM ARCHITECTURE               │
                                  └────────────────────────────────────────────────────────┘

    [ Raw Banking Data ] ───► [ Feature Engineering Pipeline ] ───► [ Tuned XGBoost Classifier ]
                                 • ZeroBalance Flag                     • scale_pos_weight: 3.91
                                 • ProductActivity Interaction          • ROC-AUC: 0.870
                                 • BalanceSalaryRatio (+Cap)            • Optimal Cutoff: 0.655
                                 • TenureAgeRatio (+Smoothing)
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
       ┌───────────────────────────────┐               ┌───────────────────────────────┐
       │   FastAPI Serving Service     │               │  Streamlit Business Cockpit   │
       │   (Production REST Backend)   │               │   (Stakeholder Intelligence)  │
       │                               │               │                               │
       │ • Pydantic Schema Validation  │               │ • Interactive Profile Sliders │
       │ • Real-time /predict (JSON)   │               │ • Live Risk Speedometer Gauge │
       │ • High-throughput /batch      │               │ • Local TreeSHAP Waterfall    │
       │ • Latency < 15ms              │               │ • Automated Retention Advice  │
       └───────────────────────────────┘               └───────────────────────────────┘
```

---

## 📊 Model Progression & Benchmark Performance

| Model Candidate | ROC-AUC | Churn Precision (1) | Churn Recall (1) | Churn F1-Score | Overall Accuracy | Business Takeaway |
|---|---|---|---|---|---|---|
| **Logistic Regression** | `0.777` | 38.0% | 70.0% | 0.49 | 71.0% | High false alarm rate (wastes marketing budget). |
| **Random Forest (Baseline)** | `0.842` | 78.0% | 45.0% | 0.57 | 86.0% | High precision, but misses 55% of actual churners. |
| **Random Forest (Tuned)** | `0.866` | 58.0% | 69.0% | 0.63 | 0.83% | Solid non-linear classification baseline. |
| **XGBoost (Baseline)** | `0.862` | 56.0% | 72.0% | 0.63 | 83.0% | Strong gradient-boosted tree performance. |
| **XGBoost (Tuned @ 0.50 Threshold)** | `0.870` | 52.0% | 76.0% | 0.62 | 81.0% | High recall, but default threshold yields too many false positives. |
| **🏆 XGBoost (Tuned @ 0.655 Calibrated Cutoff)** | **`0.870`** | **70.2%** | **60.2%** | **0.65** | **87.0%** | **Optimal balance: High accuracy, low false alarms, reliable churn capture.** |

---

## 💡 Key Machine Learning Innovations

### 1. Domain-Driven Feature Engineering
* **`ZeroBalance` Flag**: Isolates the 36.17% of customers with $0 deposits as a discrete behavioral cohort.
* **`ProductActivity`**: Multiplies `NumOfProducts * IsActiveMember` to identify high-risk inactive multi-product accounts (churn jumps from 7.5% for 2 products up to 82.7% for 3 products and 100% for 4 products).
* **`BalanceSalaryRatio`**: Computes wealth-to-income stickiness with defensive `+1` smoothing and 99th percentile clipping (`35.4762`) to prevent outlier coefficient distortion.
* **`TenureAgeRatio`**: Normalizes relationship length against age with `+0.5` additive smoothing to handle zero-tenure accounts.

### 2. Hypothesis-Driven Feature Pruning
Evaluated all features via **Point-Biserial Correlation ($r_{pb}$)** against the binary churn target ($H_0$: no linear association). Confidently pruned non-significant features:
* `HasCrCard` ($p = 0.4754$): Ubiquitous across 70.5% of accounts; zero predictive power.
* `Tenure` ($p = 0.1615$): Superseded by `TenureAgeRatio` ($p = 0.0000$).
* `EstimatedSalary` ($p = 0.2264$): Superseded by `BalanceSalaryRatio` ($p = 0.0000$).

### 3. Explainable AI (TreeSHAP)
Integrated TreeSHAP to calculate exact additive log-odds contributions for every prediction, allowing relationship managers to see *why* an individual customer is at risk.

---

## 📂 Repository Structure

```
.
├── 📂 data/
│   ├── Churn_Modelling.csv                  # Raw bank customer dataset (10,000 records)
│   └── churn_cleaned.csv                    # Cleaned & engineered dataset (13 columns)
│
├── 📂 notebooks/                            # Sequential Exploratory Notebooks
│   ├── 01_Exploratory_Data_Analysis.ipynb   # Data quality, distributions, and churn patterns
│   ├── 02_Feature_Engineering_and_Preprocessing.ipynb # Transformations, encodings, p-value selection
│   └── 03_Model_Training_and_Evaluation.ipynb # Baseline, Tuning, Thresholding, Master Table
│
├── 📂 src/                                  # Production ML Python Package
│   ├── __init__.py
│   ├── config.py                            # Centralized parameters, constants, and paths
│   ├── pipeline.py                          # Sklearn-compatible ChurnFeaturePipeline transformer
│   ├── train.py                             # Automated model training & artifact serialization
│   └── predict.py                           # Inference service & local SHAP explainability engine
│
├── 📂 artifacts/                            # Serialized Production Artifacts
│   ├── model.joblib                         # Tuned XGBoost model
│   ├── preprocessor.joblib                  # Fitted pipeline & encoders
│   ├── shap_explainer.joblib                # TreeSHAP explainer object
│   └── threshold_config.json                # Calibrated decision threshold metadata
│
├── 📂 api/                                  # FastAPI Serving Layer
│   ├── __init__.py
│   ├── main.py                              # REST API routes (/health, /predict, /predict/batch)
│   └── schemas.py                           # Pydantic data validation schemas
│
├── 📂 app/                                  # Streamlit Stakeholder Dashboard
│   └── dashboard.py                         # Interactive cockpit with sliders, gauge, SHAP plots
│
├── 📂 tests/                                # Automated Testing Suite
│   ├── __init__.py
│   ├── test_pipeline.py                     # Unit tests for math, smoothing, and clipping
│   └── test_api.py                          # API endpoint integration tests
│
├── Dockerfile                               # Production Docker container specification
├── requirements.txt                         # Pinned production dependencies
└── README.md                                # Project documentation
```

---

## 🚀 Quickstart & How to Run

### 1. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/thejas-06/Bank-Customer-Churn-Prediction.git
cd Bank-Customer-Churn-Prediction

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train and Serialize Model Artifacts
```bash
python -m src.train
```

### 3. Run the FastAPI REST Microservice
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
* **Interactive Swagger UI**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.

#### Example API Request (`POST /predict`):
```json
{
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
}
```

### 4. Launch the Streamlit Business Dashboard
```bash
streamlit run app/dashboard.py
```
* Open [http://localhost:8501](http://localhost:8501) to interact with the diagnostic sliders and SHAP waterfall chart.

### 5. Run Automated Unit & Integration Tests
```bash
pytest tests/ -v
```

### 6. Run via Docker
```bash
# Build Docker image
docker build -t churn-prediction-system .

# Run container (Exposes FastAPI on port 8000)
docker run -p 8000:8000 churn-prediction-system
```

---

## 📜 Resume Ready Highlights

* **End-to-End ML Architecture**: Built a production-grade churn classification system on 10,000 customer records, engineering 5 custom domain features and pruning noise via Point-Biserial significance testing ($p < 0.05$).
* **Optimized Decision Thresholding**: Tuned an XGBoost classifier to **0.870 ROC-AUC** and calibrated the decision boundary to **0.655**, boosting churn precision to **70.2%** and overall accuracy to **87.0%**.
* **Explainable AI (XAI)**: Implemented **TreeSHAP** to surface top churn drivers per customer and generate automated retention recommendations.
* **Production Serving & MLOps**: Packaged inference into a containerized **FastAPI** microservice with Pydantic validation, built a **Streamlit** cockpit, and enforced code reliability with **Pytest**.
