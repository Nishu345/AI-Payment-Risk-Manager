import os
import sys

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


# ============================================================
# IMPORT RISK ENGINE
# ============================================================

try:
    from risk_engine import RiskEngine
except ImportError as error:
    st.error(
        f"Could not load risk engine: {error}"
    )
    st.stop()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Payment Risk Manager",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .risk-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        text-align: center;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "transactions.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "fraud_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_transaction_data():

    if not os.path.exists(DATA_PATH):
        return None

    return pd.read_csv(
        DATA_PATH
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_risk_engine():

    if not os.path.exists(MODEL_PATH):
        return None

    return RiskEngine(
        MODEL_PATH
    )


df = load_transaction_data()

risk_engine = load_risk_engine()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '💳 AI PAYMENT RISK MANAGER'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered transaction risk analysis and '
    'fraud alert prototype'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SYSTEM CHECK
# ============================================================

if df is None:

    st.error(
        "Transaction dataset was not found."
    )

    st.info(
        "Please run: python src/data_generator.py"
    )

    st.stop()


if risk_engine is None:

    st.error(
        "Trained ML model was not found."
    )

    st.info(
        "Please run: python src/train_model.py"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ System Information"
    )

    st.write(
        "AI Payment Risk Manager"
    )

    st.write(
        "Prototype Version 1.0"
    )

    st.divider()

    st.subheader(
        "Risk Thresholds"
    )

    st.write(
        "🟢 0–30 → LOW RISK"
    )

    st.write(
        "🟡 31–70 → MEDIUM RISK"
    )

    st.write(
        "🔴 71–100 → HIGH RISK"
    )

    st.divider()

    st.caption(
        "This application is an academic/demo "
        "prototype and is not a production "
        "fraud detection service."
    )


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_transactions = len(df)

fraud_count = int(
    df["fraud_label"].sum()
)

fraud_rate = (
    fraud_count /
    total_transactions *
    100
)

average_transaction = (
    df["amount"].mean()
)


# ============================================================
# ESTIMATED RISK SCORE FOR DASHBOARD
# ============================================================

def calculate_dashboard_risk(row):

    score = 0

    amount = row["amount"]
    average_amount = (
        row["customer_avg_transaction_amount"]
    )

    amount_ratio = (
        amount /
        max(average_amount, 1)
    )

    if amount > 100000:
        score += 25

    elif amount > 50000:
        score += 18

    elif amount_ratio > 4:
        score += 18

    elif amount_ratio > 2.5:
        score += 10

    if row["is_new_device"] == 1:
        score += 15

    if row["location_mismatch"] == 1:
        score += 20

    if row["previous_failed_transactions"] >= 4:
        score += 15

    elif row["previous_failed_transactions"] >= 2:
        score += 8

    if row["transactions_last_24h"] >= 10:
        score += 15

    elif row["transactions_last_24h"] >= 6:
        score += 8

    if row["account_age_days"] < 30:
        score += 8

    elif row["account_age_days"] < 90:
        score += 4

    if 0 <= row["transaction_hour"] <= 4:
        score += 7

    return min(
        score,
        100
    )


dashboard_df = df.copy()

dashboard_df["risk_score"] = (
    dashboard_df.apply(
        calculate_dashboard_risk,
        axis=1
    )
)


dashboard_df["risk_level"] = (
    dashboard_df["risk_score"]
    .apply(
        lambda score:
        "LOW RISK"
        if score <= 30
        else
        "MEDIUM RISK"
        if score <= 70
        else
        "HIGH RISK"
    )
)


average_risk_score = (
    dashboard_df["risk_score"].mean()
)


# ============================================================
# KPI CARDS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 Transaction Overview'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with col2:

    st.metric(
        "Fraud Detected",
        f"{fraud_count:,}"
    )

with col3:

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

with col4:

    st.metric(
        "Average Risk Score",
        f"{average_risk_score:.1f}/100"
    )


# ============================================================
# ANALYTICS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📈 Transaction Analytics'
    '</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Chart 1 — Genuine vs Fraud
# ------------------------------------------------------------

fraud_chart_data = pd.DataFrame(
    {
        "Transaction Type": [
            "Genuine",
            "Fraud"
        ],
        "Count": [
            int(
                (df["fraud_label"] == 0).sum()
            ),
            int(
                (df["fraud_label"] == 1).sum()
            )
        ]
    }
)

fig_fraud = px.bar(
    fraud_chart_data,
    x="Transaction Type",
    y="Count",
    title="Genuine vs Fraud Transactions",
    text="Count"
)

fig_fraud.update_layout(
    showlegend=False
)

fig_fraud.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_fraud,
    use_container_width=True
)


# ------------------------------------------------------------
# Chart 2 — Payment Method
# ------------------------------------------------------------

payment_data = (
    df["payment_method"]
    .value_counts()
    .reset_index()
)

payment_data.columns = [
    "Payment Method",
    "Transactions"
]

fig_payment = px.bar(
    payment_data,
    x="Payment Method",
    y="Transactions",
    title="Transactions by Payment Method",
    text="Transactions"
)

st.plotly_chart(
    fig_payment,
    use_container_width=True
)


# ------------------------------------------------------------
# Chart 3 — Risk Distribution
# ------------------------------------------------------------

risk_distribution = (
    dashboard_df["risk_level"]
    .value_counts()
    .reset_index()
)

risk_distribution.columns = [
    "Risk Level",
    "Transactions"
]

fig_risk = px.pie(
    risk_distribution,
    names="Risk Level",
    values="Transactions",
    title="Risk Distribution"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)


# ------------------------------------------------------------
# Chart 4 — Transaction Amount
# ------------------------------------------------------------

fig_amount = px.histogram(
    df,
    x="amount",
    nbins=50,
    title="Transaction Amount Distribution"
)

st.plotly_chart(
    fig_amount,
    use_container_width=True
)


# ------------------------------------------------------------
# Chart 5 — Device Type
# ------------------------------------------------------------

device_data = (
    df.groupby("device_type")["fraud_label"]
    .sum()
    .reset_index()
)

device_data.columns = [
    "Device Type",
    "Fraud Transactions"
]

fig_device = px.bar(
    device_data,
    x="Device Type",
    y="Fraud Transactions",
    title="Fraud by Device Type",
    text="Fraud Transactions"
)

st.plotly_chart(
    fig_device,
    use_container_width=True
)


# ------------------------------------------------------------
# Chart 6 — Location
# ------------------------------------------------------------

location_data = (
    df.groupby("location")["fraud_label"]
    .sum()
    .reset_index()
)

location_data.columns = [
    "Location",
    "Fraud Transactions"
]

location_data = location_data.sort_values(
    "Fraud Transactions",
    ascending=False
)

fig_location = px.bar(
    location_data,
    x="Location",
    y="Fraud Transactions",
    title="Fraud by Location",
    text="Fraud Transactions"
)

st.plotly_chart(
    fig_location,
    use_container_width=True
)


# ------------------------------------------------------------
# Chart 7 — Transaction Risk Trend
# ------------------------------------------------------------

trend_data = (
    dashboard_df
    .groupby("transaction_hour")["risk_score"]
    .mean()
    .reset_index()
)

fig_trend = px.line(
    trend_data,
    x="transaction_hour",
    y="risk_score",
    markers=True,
    title="Average Risk Score by Transaction Hour"
)

fig_trend.update_layout(
    xaxis_title="Transaction Hour",
    yaxis_title="Average Risk Score"
)

st.plotly_chart(
    fig_trend,
    use_container_width=True
)


# ============================================================
# TRANSACTION ANALYZER
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔎 Analyze New Transaction'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter transaction details below and "
    "click ANALYZE TRANSACTION to calculate "
    "the risk."
)


# ------------------------------------------------------------
# Input columns
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=1.0,
        max_value=500000.0,
        value=2500.0,
        step=100.0
    )

    transaction_hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=14
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "UPI",
            "Credit Card",
            "Debit Card",
            "Wallet",
            "Net Banking"
        ]
    )

    location = st.selectbox(
        "Location",
        [
            "Delhi",
            "Mumbai",
            "Bengaluru",
            "Hyderabad",
            "Chennai",
            "Pune",
            "Kolkata",
            "Noida",
            "Gurugram",
            "Jaipur"
        ]
    )


with col2:

    device_type = st.selectbox(
        "Device Type",
        [
            "Android",
            "iOS",
            "Web"
        ]
    )

    is_new_device = st.selectbox(
        "Is this a new device?",
        [
            "No",
            "Yes"
        ]
    )

    location_mismatch = st.selectbox(
        "Location mismatch?",
        [
            "No",
            "Yes"
        ]
    )

    previous_failed_transactions = st.number_input(
        "Previous Failed Transactions",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )


with col3:

    transactions_last_24h = st.number_input(
        "Transactions in Last 24 Hours",
        min_value=1,
        max_value=50,
        value=3,
        step=1
    )

    account_age_days = st.number_input(
        "Account Age (Days)",
        min_value=1,
        max_value=5000,
        value=365,
        step=1
    )

    customer_avg_transaction_amount = st.number_input(
        "Customer Average Transaction Amount (₹)",
        min_value=1.0,
        max_value=100000.0,
        value=2000.0,
        step=100.0
    )

    is_weekend = st.selectbox(
        "Weekend?",
        [
            "No",
            "Yes"
        ]
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 ANALYZE TRANSACTION",
    type="primary",
    use_container_width=True
)


if analyze_button:

    # --------------------------------------------------------
    # Safety validation
    # --------------------------------------------------------

    if amount <= 0:

        st.error(
            "Transaction amount must be greater than zero."
        )

        st.stop()

    if customer_avg_transaction_amount <= 0:

        st.error(
            "Customer average transaction amount "
            "must be greater than zero."
        )

        st.stop()

    # --------------------------------------------------------
    # Convert inputs
    # --------------------------------------------------------

    transaction = {

        "amount":
            float(amount),

        "transaction_hour":
            int(transaction_hour),

        "payment_method":
            payment_method,

        "location":
            location,

        "device_type":
            device_type,

        "is_new_device":
            1
            if is_new_device == "Yes"
            else 0,

        "location_mismatch":
            1
            if location_mismatch == "Yes"
            else 0,

        "previous_failed_transactions":
            int(previous_failed_transactions),

        "transactions_last_24h":
            int(transactions_last_24h),

        "account_age_days":
            int(account_age_days),

        "customer_avg_transaction_amount":
            float(
                customer_avg_transaction_amount
            ),

        "is_weekend":
            1
            if is_weekend == "Yes"
            else 0
    }

    # --------------------------------------------------------
    # Run risk analysis
    # --------------------------------------------------------

    try:

        result = risk_engine.analyze_transaction(
            transaction
        )

    except Exception as error:

        st.error(
            "Unable to analyze this transaction."
        )

        st.exception(error)

        st.stop()

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🚨 Risk Assessment Result"
    )

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    with result_col1:

        st.metric(
            "Risk Score",
            f"{result['risk_score']}/100"
        )

    with result_col2:

        st.metric(
            "Risk Level",
            result["risk_level"]
        )

    with result_col3:

        st.metric(
            "Fraud Probability",
            f"{result['fraud_probability']}%"
        )

    # --------------------------------------------------------
    # Risk level message
    # --------------------------------------------------------

    if result["risk_level"] == "LOW RISK":

        st.success(
            "LOW RISK — Transaction appears relatively safe."
        )

    elif result["risk_level"] == "MEDIUM RISK":

        st.warning(
            "MEDIUM RISK — Additional verification "
            "is recommended."
        )

    else:

        st.error(
            "HIGH RISK — Additional authentication "
            "and review are recommended."
        )

    # --------------------------------------------------------
    # Risk factors
    # --------------------------------------------------------

    st.subheader(
        "⚠️ Risk Factors"
    )

    for factor in result["risk_factors"]:

        st.write(
            f"• {factor}"
        )

    # --------------------------------------------------------
    # Recommended action
    # --------------------------------------------------------

    st.subheader(
        "🛡️ Recommended Action"
    )

    st.info(
        result["recommended_action"]
    )

    # --------------------------------------------------------
    # Transaction details
    # --------------------------------------------------------

    with st.expander(
        "View Transaction Details"
    ):

        transaction_display = pd.DataFrame(
            [
                transaction
            ]
        ).T

        transaction_display.columns = [
            "Value"
        ]

        st.dataframe(
            transaction_display,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Payment Risk Manager | "
    "Synthetic-data academic/demo prototype | "
    "Not a production fraud detection system"
)