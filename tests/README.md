# 🧪 Automated Testing Suite (`tests/`)

This directory contains the automated unit and integration tests built using **Pytest** and FastAPI's **TestClient**.

---

## 📑 Test Suite Inventory

| Test File | Test Type | Focus Area | Tests |
|---|---|---|---|
| **`test_pipeline.py`** | Unit Tests | Mathematical correctness of `ChurnFeaturePipeline`, zero-tenure smoothing, outlier clipping, and encodings. | 6 Tests |
| **`test_api.py`** | Integration Tests | FastAPI HTTP endpoints (`/health`, `/predict`, `/predict/batch`) and Pydantic validation error handling. | 5 Tests |

**Total Tests**: **11 Tests** (100% Pass Rate).

---

## 🔍 Detailed Test Coverage

### `test_pipeline.py` (Feature Engineering Unit Tests)
1. **`test_pipeline_output_shape_and_columns`**: Confirms that transformation outputs a clean `(N, 12)` matrix with exact column order.
2. **`test_zero_balance_feature`**: Asserts `Balance == 0` flags as `1`, and `Balance > 0` flags as `0`.
3. **`test_product_activity_interaction`**: Validates interaction: `NumOfProducts * IsActiveMember`.
4. **`test_zero_tenure_smoothing`**: Verifies `(0 + 0.5) / Age` additive smoothing for zero-tenure accounts.
5. **`test_balance_salary_ratio_outlier_capping`**: Asserts extreme ratios are clipped at `35.4762`.
6. **`test_categorical_encoding`**: Verifies binary `Gender` mapping and `Geography` dummy variable generation.

---

### `test_api.py` (FastAPI Integration Tests)
1. **`test_health_endpoint`**: Asserts `GET /health` returns `200 OK` and `model_ready: True`.
2. **`test_predict_single_valid_customer`**: Asserts `POST /predict` returns `200 OK`, valid probability range $[0.0, 1.0]$, boolean flag, and SHAP factors.
3. **`test_predict_invalid_credit_score`**: Sends out-of-range `CreditScore: 250`, asserts `422 Unprocessable Entity`.
4. **`test_predict_invalid_geography`**: Sends invalid country (`"Canada"`), asserts `422 Unprocessable Entity`.
5. **`test_predict_batch`**: Asserts `POST /predict/batch` returns `200 OK` with aggregate cohort statistics.

---

## 🚀 How to Run Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run only pipeline unit tests
pytest tests/test_pipeline.py -v

# Run only API integration tests
pytest tests/test_api.py -v
```
