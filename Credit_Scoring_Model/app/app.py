import streamlit as st
import sys
import os
import pandas as pd

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_credit_score

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide"
)

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------


st.markdown("""
<style>

input {
    caret-color: black !important;
}

            
/* General text */
body,
p,
label,
span,
div{
    color: #1f2937;
}

/* Info box text */
div[data-testid="stAlert"]{

    background:white;

    border-left:6px solid #0B3C91;

    border-radius:12px;

    color:#222;

}


/* Main Background */
.stApp{

    background:linear-gradient(135deg,#eef4ff,#ffffff);

}

/* Title */
.title{

    text-align:center;

    font-size:44px;

    font-weight:bold;

    background:linear-gradient(90deg,#0B3C91,#00AEEF);

    -webkit-background-clip:text;


}

/* Subtitle */
.subtitle{
    text-align:center;
    color:#555;
    font-size:18px;
    margin-bottom:20px;
}

/* ********************************
PREDICT BUTTON
******************************** */

.stButton > button{

    width:100% !important;
    height:60px !important;
    background:linear-gradient(90deg,#0B3C91,#1E88E5) !important;
    color:white !important;
    border:none !important;
    border-radius:15px !important;
    font-size:20px !important;
    font-weight:bold !important;

}

.stButton > button:hover{

    background:linear-gradient(90deg,#1565C0,#42A5F5) !important;
    color:white !important;

}

/* Number Input */

/* ******************************
INPUT BOXES
******************************* */

div[data-baseweb="input"] > div{

    background-color:#ffffff !important;

    border:2px solid #d6e4ff !important;

    border-radius:10px !important;

}

div[data-baseweb="input"] input{

    background-color:#ffffff !important;

    color:#0f172a !important;

    font-weight:600;

}

button[aria-label="Increment"],
button[aria-label="Decrement"]{

    background:#ffffff !important;

    color:#0B3C91 !important;

}

/* Metric Cards */

div[data-testid="metric-container"]{

    background:white;
    color:#0B3C91;
    border-radius:12px;
    padding:20px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.10);
    
}

/* Sidebar */

section[data-testid="stSidebar"]{

    background:#0B3C91;

}

section[data-testid="stSidebar"] *{

    color:white;

}

/* ==========================================
   CUSTOMER SUMMARY TABLE
========================================== */

table {

    border-collapse: collapse !important;

    width: 100% !important;

    background: white !important;

    border-radius: 12px !important;

    overflow: hidden !important;

}

thead tr {

    background: #0B3C91 !important;
    color: white !important;
            

}

thead th {

    color: white !important;

    text-align: center !important;

    font-size: 16px !important;

    padding: 12px !important;

}

tbody td {

    background: #F8FBFF !important;

    color: #1F2937 !important;

    text-align: center !important;

    padding: 10px !important;

    border-bottom: 1px solid #E5E7EB !important;

}

tbody tr:nth-child(even) td {

    background: #EEF4FF !important;

}            

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.markdown(
    "<div class='title'>💳 Credit Scoring Model</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict Customer Creditworthiness using Machine Learning</div>",
    unsafe_allow_html=True
)

st.write("---")

# ---------------------------------------------------
# Project Information
# ---------------------------------------------------

st.info("""
**Model Used:** Random Forest

**Dataset:** Give Me Some Credit

**Goal:** Predict whether a customer is likely to experience serious financial distress within the next two years.
""")

# ---------------------------------------------------
# Model Performance
# ---------------------------------------------------

st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Accuracy", "87.83%")

with c2:
    st.metric("Recall", "52.30%")

with c3:
    st.metric("ROC-AUC", "83.10%")

st.write("---")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.header("🏦 Project Information")

st.sidebar.write("""
## Objective

Predict customer creditworthiness using historical financial information.

## Algorithms

- Logistic Regression
- Decision Tree
- Random Forest ✅

## Dataset

Give Me Some Credit

""")

# ---------------------------------------------------
# Input Fields
# ---------------------------------------------------

st.markdown(
"<h2 style='text-align:center;'>📝 Customer Information</h2>",
unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    revolving = st.number_input(
        "Revolving Utilization",
        min_value=0.0,
        value=0.50
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    past_due_30 = st.number_input(
        "30-59 Days Past Due",
        min_value=0,
        value=0
    )

    debt_ratio = st.number_input(
        "Debt Ratio",
        min_value=0.0,
        value=0.40
    )

    income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=5000.0
    )

with col2:

    open_credit = st.number_input(
        "Open Credit Lines",
        min_value=0,
        value=8
    )

    late_90 = st.number_input(
        "90 Days Late",
        min_value=0,
        value=0
    )

    real_estate = st.number_input(
        "Real Estate Loans",
        min_value=0,
        value=1
    )

    late_60 = st.number_input(
        "60-89 Days Past Due",
        min_value=0,
        value=0
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        value=2
    )

st.write("")

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("🔍 Predict Credit Status"):

    customer = {

        "RevolvingUtilizationOfUnsecuredLines": revolving,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": past_due_30,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": open_credit,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60,
        "NumberOfDependents": dependents

    }

    prediction, confidence = predict_credit_score(customer)

    st.write("---")

    if prediction == "Good Credit":

        st.success("✅ Customer is Financially Stable")

    else:

        st.error("⚠ High Credit Risk Detected")

    st.metric(
        "Prediction Confidence",
        f"{confidence*100:.2f}%"
    )

    st.progress(float(confidence))

    # -------------------------------------
    # Confidence Level
    # -------------------------------------

    if confidence >= 0.90:
        level = "Very High"

    elif confidence >= 0.75:
        level = "High"

    elif confidence >= 0.60:
        level = "Moderate"

    else:
        level = "Low"

    st.info(f"Confidence Level: **{level}**")

    # -------------------------------------
    # Customer Summary
    # -------------------------------------

    st.subheader("📋 Customer Summary")

    summary = pd.DataFrame(
        customer.items(),
        columns=["Feature", "Value"]
    )

    styled_summary = summary.style.set_properties(**{
        "color": "white",
        "background-color": "#1E293B",
        "text-align": "center"
    })

    st.table(styled_summary)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.write("---")

st.markdown("""
---
<div style='text-align:center;color:gray'>

💳 Credit Scoring Model

Built with ❤️ using Python, Scikit-learn and Streamlit

CodeAlpha Machine Learning Internship

Python • Scikit-learn • Streamlit

© 2026

</div>
""", unsafe_allow_html=True)