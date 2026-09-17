# 📊 Streamlit Business Stakeholder Cockpit (`app/`)

This directory contains the user-facing web dashboard built with **Streamlit**, **Plotly**, and **TreeSHAP** for non-technical banking relationship managers and customer retention teams.

---

## 📑 Application Features & Workflows

### 1. 🎯 Single Customer Diagnostic
* **Interactive Controls**: Sliders and dropdowns for demographic, financial, and product holding parameters.
* **Plotly Risk Speedometer**: Real-time gauge chart color-coded into:
  * 🟢 Green ($0\% - 40\%$): Low Risk / Retained
  * 🟡 Yellow ($40\% - 70\%$): Moderate Risk
  * 🔴 Red ($70\% - 100\%$): Severe Churn Danger
  * Bold red marker indicating the calibrated **`65.5%`** decision threshold.
* **Local TreeSHAP Attribution Bar Chart**: Horizontal bar chart showing exact feature-level log-odds contributions:
  * 🔴 **Red Bars**: Push customer toward leaving (e.g. `NumOfProducts=3`, `Age=52`).
  * 🟢 **Green Bars**: Protect customer retention (e.g. `IsActiveMember=1`, `Balance=$120k`).
* **Actionable Retention Strategy**: Rule-based banking playbook recommendations (e.g., fee waivers, dedicated relationship manager assignment).

---

### 2. 📁 Batch CSV Scoring
* **Drag-and-Drop Uploader**: Upload raw customer spreadsheets with thousands of rows.
* **KPI Metrics**: Real-time summary cards displaying Total Evaluated, Flagged Churn Risk count, and Retained count.
* **Cohort Breakdown**: Interactive Plotly pie chart visualizing cohort risk distribution.
* **1-Click Export**: Download scored CSV (`scored_bank_churn_predictions.csv`) with predicted probabilities and risk tiers appended.

---

## 🚀 How to Run Locally

```bash
streamlit run app/dashboard.py
```

* **Dashboard URL**: Open [http://localhost:8501](http://localhost:8501) in your browser.
