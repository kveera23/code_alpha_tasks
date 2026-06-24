import streamlit as st
import pandas as pd

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.predict import predict_heart_disease

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
input[type="number"]{
        caret-color: black !important;    
            }            
            
</style>""",unsafe_allow_html=True)

st.markdown("""
<style>

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: white !important;
}
            
/* Number Input Background */
.stNumberInput > div > div {
    background-color: white !important;
    border-radius: 10px !important;
}

/* Plus and Minus Buttons */
.stNumberInput button {
    background-color: #E3F2FD !important;
    color: #003366 !important;
    border: none !important;
}

/* Hover Effect */
.stNumberInput button:hover {
    background-color: #BBDEFB !important;
}

/* Number Input Text */
.stNumberInput input {
    color: #003366 !important;
    background-color: white !important;
}
            
            
/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #F8FBFF,
        #EEF6FF,
        #DDEEFF
    );
}

/* Main container */
.block-container {
    padding-top: 2rem;
}

/* Title */
h1 {
    text-align: center;
    color: #003366 !important;
    font-weight: 800;
    font-size: 3rem;
}

/* Input boxes */
.stNumberInput, .stSelectbox {
    background-color: rgba(255,255,255,0.8);
    border-radius: 10px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(
        90deg,
        #1976D2,
        #64B5F6
    );
    color: white;
    border: none;
}

/* Metric Card */
div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.15);            
}

label {
    color: black !important;
    font-weight: 700 !important;
}
            
div[data-testid="stVerticalBlock"] {
    border-radius: 12px;
}


/* Input Fields */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: white !important;
    color: #003366 !important;
    border-radius: 10px !important;
}
            
div[data-testid="stAlert"] {
    background-color: #E3F2FD !important;
    color: #003366 !important;
    border-radius: 12px;
}

/* Select Boxes */
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: #003366 !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1565C0,
        #1E88E5
    );
}

     
/* Metric Card */
div[data-testid="stMetric"] {
    background: white !important;
    color: #003366 !important;
    padding: 20px;
    border-radius: 15px;
    border-left: 6px solid #1976D2;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

/* Metric Value */
div[data-testid="stMetricValue"] {
    color: #1976D2 !important;
    font-size: 2rem !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:

    st.markdown(
    """
    <h2 style='color:white;'>
        ❤️ Project Dashboard
    </h2>
    """,
    unsafe_allow_html=True
    )

    st.markdown("""
    ### Heart Disease Prediction

    **Algorithm:** Random Forest

    **Dataset:** UCI Heart Disease

    **Accuracy:** 88.52%

    **ROC-AUC:** 95.18%

    ---
    """)

    st.markdown("""
    <div style="
    background:white;
    padding:11px 5px 0px 25px;
    border-radius:12px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
    display:flex;
    align-items:center;
    height:60px;
    ">

    <span style="
    display:inline-block;
    width:30px;
    height:30px;
    line-height:30px;
    text-align:center;
    border-radius:50%;
    background:#1976D2;
    color:white;
    font-weight:bold;
    font-size:18px;
    margin-right:12px;
    ">
    ✓
    </span>

    <span style="
    color:#1E88E5;
    font-size:18px;
    font-weight:600;
    ">
    Model Status: Active
    </span>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.write("Developed By")

    st.write("**Veeranjaneyulu Kopparapu**")

# ==========================================
# HEADER
# ==========================================

st.markdown(
    """
    <h1 style='text-align:center;
               color:#003366;
               font-size:48px;'>
        ❤️ Heart Disease Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

st.info(
    """
    This application predicts the likelihood of heart disease
    using a Random Forest Machine Learning model trained on
    the UCI Heart Disease dataset.
    """
)

st.write("### Enter Patient Details")

# ==========================================
# INPUTS (TWO COLUMN LAYOUT)
# ==========================================

# Chest Pain Type
cp_options = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}

# Resting ECG
restecg_options = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

# Slope
slope_options = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

# Thalassemia Type
thal_options = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2,
    "Unknown": 3
}

col1, col2 = st.columns(2)

# ---------- LEFT COLUMN ----------
with col1:

    age = age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

    sex_option = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    cp_label = st.selectbox(
        "Chest Pain Type",
        list(cp_options.keys())
    )

    chol = chol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=100,
            max_value=400,
            value=200
        )

    restecg_label = st.selectbox(
        "Resting ECG",
        list(restecg_options.keys())
    )

    exang_option = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

    slope_label = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        list(slope_options.keys())
    )

# ---------- RIGHT COLUMN ----------
with col2:

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50,
        max_value=250,
        value=120
    )

    fbs_option = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3]
    )

    thal_label = st.selectbox(
        "Thalassemia Type",
        list(thal_options.keys())
    )

# ==========================================
# CONVERSIONS
# ==========================================

sex = 0 if sex_option == "Female" else 1

fbs = 0 if fbs_option == "No" else 1

exang = 0 if exang_option == "No" else 1

cp = cp_options[cp_label]
restecg = restecg_options[restecg_label]
slope = slope_options[slope_label]
thal = thal_options[thal_label]



# ==========================================
# PREDICTION
# ==========================================

if st.button("🔍 Predict Heart Disease Risk"):

    input_data = pd.DataFrame([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]], columns=[
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ])

    prediction, probability = predict_heart_disease(input_data)

    st.subheader("Prediction Result")

    st.metric(
        label="Heart Disease Risk Probability",
        value=f"{probability:.2%}"
    )

    if probability < 0.30:

        risk_level = "🟢 Low Risk"
        st.markdown(
            """
            <div style="
            background:#E8F5E9;
            color:#1B5E20;
            padding:18px;
            border-radius:12px;
            border-left:6px solid #4CAF50;
            font-weight:600;">
            ✅ Patient shows a low probability of heart disease based on the provided clinical parameters.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif probability < 0.70:

        risk_level = "🟡 Moderate Risk"

        st.markdown(
            """
            <div style="
            background:#FFF8E1;
            color:#E65100;
            padding:18px;
            border-radius:12px;
            border-left:6px solid #FF9800;
            font-weight:600;">
            ⚠️ Patient shows a moderate probability of heart disease. Further medical evaluation may be recommended.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        risk_level = "🔴 High Risk"

        st.markdown(
            """
            <div style="
            background:#FFEBEE;
            color:#B71C1C;
            padding:18px;
            border-radius:12px;
            border-left:6px solid #F44336;
            font-weight:600;">
            🚨 Patient shows a high probability of heart disease. Medical consultation is strongly recommended.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
            f"""
            <div style="
            background:white;
            color:#003366;
            padding:15px;
            border-radius:12px;
            border-left:6px solid #1976D2;
            font-weight:bold;">
            Risk Level: {risk_level}
            </div>
            """,
            unsafe_allow_html=True
        )

    if prediction == 1:
        st.markdown(
            """
            <div style="
            background:#FFEBEE;
            color:#B71C1C;
            padding:18px;
            border-radius:12px;
            border-left:6px solid #F44336;
            font-weight:bold;">
            ⚠️ Heart Disease Detected
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
            background:#E8F5E9;
            color:#1B5E20;
            padding:18px;
            border-radius:12px;
            border-left:6px solid #4CAF50;
            font-weight:bold;">
            ✅ No Heart Disease Detected
            </div>
            """,
            unsafe_allow_html=True
        )



st.markdown("---")

st.subheader("📊 Model Insights")

tab1, tab2, tab3 = st.tabs([
    "Feature Importance",
    "ROC Curve",
    "Confusion Matrix"
])

with tab1:
    st.image(
        os.path.join(BASE_DIR, "reports", "images", "feature_importance.png"),
        use_container_width=True
    )

with tab2:
    st.image(
        os.path.join(BASE_DIR, "reports", "images", "roc_curve.png"),
        use_container_width=True
    )

with tab3:
    st.image(
        os.path.join(BASE_DIR, "reports", "images", "confusion_matrix.png"),
        use_container_width=True
    )



# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Developed as part of a Machine Learning Internship Project | Heart Disease Prediction using Random Forest and UCI Heart Disease Dataset"
)