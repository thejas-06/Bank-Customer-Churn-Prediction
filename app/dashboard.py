"""
Streamlit Stakeholder Dashboard for Bank Customer Churn Prediction & Explainability.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import io
import sys
from pathlib import Path

# Add project root to Python search path so 'src' module imports seamlessly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import get_predictor
from src.config import OPTIMAL_THRESHOLD, FINAL_MODEL_FEATURES

# Page Configuration
st.set_page_config(
    page_title="Bank Churn Intelligence Suite",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, enterprise look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-high {
        color: #DC2626;
        font-weight: bold;
    }
    .risk-medium {
        color: #D97706;
        font-weight: bold;
    }
    .risk-low {
        color: #16A34A;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def render_gauge_chart(probability: float, threshold: float = OPTIMAL_THRESHOLD):
    """Renders a Plotly speedometer gauge chart for churn risk."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 36, 'color': '#1E293B'}},
        title={'text': "<b>Churn Probability Score</b>", 'font': {'size': 20, 'color': '#334155'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748B"},
            'bar': {'color': "#1E3A8A", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [0, 40], 'color': '#DCFCE7'},
                {'range': [40, 70], 'color': '#FEF3C7'},
                {'range': [70, 100], 'color': '#FEE2E2'}
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def main():
    st.markdown('<div class="main-header">🏦 Bank Customer Churn Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explainable AI (TreeSHAP) & Calibrated Risk Decision System for Retention Strategy</div>', unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Workflow:",
        ["🎯 Single Customer Diagnostic", "📁 Batch CSV Scoring"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Engine**: Tuned XGBoost")
    st.sidebar.markdown(f"**Decision Cutoff**: `{OPTIMAL_THRESHOLD:.3f}`")
    st.sidebar.markdown(f"**Target Precision**: `70.2%` | **Recall**: `60.2%`")

    predictor = get_predictor()

    # =========================================================================
    # TAB 1: Single Customer Diagnostic
    # =========================================================================
    if app_mode == "🎯 Single Customer Diagnostic":
        st.subheader("Customer Profile & Account Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographics")
            geography = st.selectbox("Country of Residence", ["France", "Germany", "Spain"], index=0)
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            age = st.slider("Customer Age", min_value=18, max_value=92, value=42, step=1)
            credit_score = st.slider("Credit Score", min_value=350, max_value=850, value=650, step=5)

        with col2:
            st.markdown("##### 💳 Financial Standing")
            balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=250000.0, value=75000.0, step=1000.0)
            estimated_salary = st.number_input("Estimated Annual Salary ($)", min_value=10.0, max_value=200000.0, value=100000.0, step=1000.0)
            has_cr_card = st.radio("Holds Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

        with col3:
            st.markdown("##### 📦 Relationship & Engagement")
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
            tenure = st.slider("Tenure (Years with Bank)", min_value=0, max_value=10, value=3, step=1)
            is_active = st.radio("Active Member Status", [1, 0], format_func=lambda x: "Active" if x == 1 else "Inactive", horizontal=True)

        st.markdown("---")

        if st.button("🚀 Evaluate Customer Churn Risk", type="primary", use_container_width=True):
            customer_data = {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active,
                "EstimatedSalary": estimated_salary
            }

            with st.spinner("Analyzing customer behavioral patterns and running TreeSHAP attribution..."):
                res = predictor.predict_single(customer_data)

            prob = res["churn_probability"]
            will_churn = res["will_churn"]
            risk_level = res["risk_level"]

            # Results Display
            st.markdown("### 📋 Diagnostic Assessment")
            res_col1, res_col2 = st.columns([1, 1.2])

            with res_col1:
                st.plotly_chart(render_gauge_chart(prob), use_container_width=True)

                if will_churn:
                    st.error(f"🚨 **HIGH RISK**: This customer is flagged to **CHURN** (Probability: {prob*100:.1f}% >= Threshold: {OPTIMAL_THRESHOLD*100:.1f}%)")
                else:
                    st.success(f"✅ **SAFE**: Customer is projected to **REMAIN** (Probability: {prob*100:.1f}% < Threshold: {OPTIMAL_THRESHOLD*100:.1f}%)")

                st.markdown(f"**Categorical Risk Tier:** `{risk_level}`")

            with res_col2:
                st.markdown("#### 💡 Tailored Retention Strategy")
                st.info(f"**Recommended Action:**\n\n{res['retention_recommendation']}")

                st.markdown("#### 🔍 Top Positive Drivers of Churn Risk")
                if res["top_risk_factors"]:
                    for factor in res["top_risk_factors"]:
                        st.markdown(f"- **`{factor['feature']}`** = `{factor['value']}` $\\rightarrow$ **+{factor['shap_impact']:.3f}** log-odds contribution")
                else:
                    st.markdown("*No dominant risk factors identified (Customer exhibits strong loyalty indicators).*")

            # SHAP Feature Impact Chart
            st.markdown("---")
            st.markdown("#### 📊 Explainable AI: Feature Attribution Breakdown (TreeSHAP)")
            
            impacts_df = pd.DataFrame(res["all_feature_impacts"])
            impacts_df = impacts_df.sort_values(by="shap_impact", ascending=True)

            colors = ['#DC2626' if x > 0 else '#16A34A' for x in impacts_df["shap_impact"]]
            
            fig_shap = go.Figure(go.Bar(
                x=impacts_df["shap_impact"],
                y=impacts_df["feature"],
                orientation='h',
                marker_color=colors,
                text=impacts_df["shap_impact"].round(3),
                textposition='auto'
            ))
            fig_shap.update_layout(
                title="<b>Local SHAP Values</b> (Red = Increases Churn Probability, Green = Protects Retention)",
                xaxis_title="SHAP Contribution to Log-Odds",
                yaxis_title="Feature",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_shap, use_container_width=True)

    # =========================================================================
    # TAB 2: Batch CSV Scoring
    # =========================================================================
    elif app_mode == "📁 Batch CSV Scoring":
        st.subheader("High-Throughput Batch Churn Processing")
        st.markdown("Upload a raw CSV containing customer records to score churn risk in bulk.")

        uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df_batch)}** customer records.")
            st.dataframe(df_batch.head(5), use_container_width=True)

            if st.button("⚡ Score Batch Records", type="primary"):
                with st.spinner("Scoring batch customer data..."):
                    scored_df = predictor.predict_batch(df_batch)

                total = len(scored_df)
                churn_count = (scored_df["Predicted_Churn"] == 1).sum()
                churn_rate = (churn_count / total) * 100

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Customers Evaluated", f"{total:,}")
                m2.metric("Flagged Churn Risk", f"{churn_count:,}", f"{churn_rate:.1f}% rate", delta_color="inverse")
                m3.metric("Safe / Retained", f"{(total - churn_count):,}")

                # Risk distribution chart
                risk_counts = scored_df["Risk_Level"].value_counts().reset_index()
                risk_counts.columns = ["Risk_Level", "Count"]
                fig_pie = px.pie(
                    risk_counts,
                    names="Risk_Level",
                    values="Count",
                    title="Cohort Risk Distribution",
                    color="Risk_Level",
                    color_discrete_map={"Low": "#16A34A", "Medium": "#D97706", "High": "#DC2626"}
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                st.subheader("Scored Customer Dataset")
                st.dataframe(scored_df.head(20), use_container_width=True)

                csv_buffer = io.StringIO()
                scored_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Scored CSV",
                    data=csv_buffer.getvalue(),
                    file_name="scored_bank_churn_predictions.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()
