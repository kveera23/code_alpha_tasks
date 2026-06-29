import streamlit as st

from app.predictor import predict_uploaded_image
from app.utils import load_css

# ==================================================
# Page Config
# ==================================================

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide"
)


with st.sidebar:

    st.markdown("## 🧠 Model Dashboard")

    st.success("🟢 Model Status : Active")

    st.divider()

    st.markdown("### 📈 Performance")

    st.markdown("""
        <div class="metric">

        <h2>94.81%</h2>

        <p>Accuracy</p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="metric">

        <h2>26</h2>

        <p>Classes</p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="metric">

        <h2>EMNIST</h2>

        <p>Dataset</p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="metric">

        <h2>CNN</h2>

        <p>Model</p>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div class="card">

    <h3>📤 Ready for Prediction</h3>

    <p style="color:#94a3b8;">
    Upload a handwritten English alphabet image to begin prediction.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### ℹ️ About")

    st.write("""
    This application recognizes
    handwritten English alphabets
    using a Convolutional Neural
    Network trained on the
    EMNIST Letters dataset.
    """)

# ==================================================
# Load CSS
# ==================================================

from src.config import CSS_PATH
load_css(CSS_PATH)

# ==================================================
# Hero Section
# ==================================================

st.markdown("""
<div class="hero">

<h1>✍️ Handwritten Character Recognition</h1>

<p>
Deep Learning • CNN • EMNIST Letters Dataset
</p>

</div>
""", unsafe_allow_html=True)

# ==================================================
# Two Column Layout
# ==================================================

left, right = st.columns([1.1, 0.9], gap="large")

# ==================================================
# Upload Section
# ==================================================

with left:
    st.markdown("""
    <div class="card">

    <h2 style="margin-bottom:5px;">
    📤 Upload Image
    </h2>

    <p style="color:#94a3b8;">
    Upload a handwritten English alphabet image (PNG, JPG or JPEG).
    </p>

    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["png","jpg","jpeg"],
        label_visibility="collapsed"
    )

# ==================================================
# Prediction
# ==================================================

if uploaded_file is not None:

    image, letter, confidence = predict_uploaded_image(uploaded_file)

    with left:

        st.markdown("""
        <div class="card">

        <h2 style="text-align:center;">
        🖼 Uploaded Image
        </h2>

        <p style="text-align:center;color:#94a3b8;">

        Preview of uploaded handwritten character

        </p>

        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(
                image,
                width=250
            )

    with right:

        st.markdown(f"""
        <div class="card">

        <h2 style="text-align:center;">
        🎯 Prediction Result
        </h2>

        <div class="prediction">

        {letter}

        </div>

        <div style="text-align:center;
        font-size:18px;
        color:#cbd5e1;">

        Predicted Character

        </div>

        <br>

        <div class="confidence">

        Confidence

        <br>

        {confidence*100:.2f}%

        </div>

        </div>
        """, unsafe_allow_html=True)


        st.markdown(f"""
            <div style="margin-top:20px;">

            <div style="
            width:100%;
            height:14px;
            background:#1e293b;
            border-radius:30px;
            overflow:hidden;
            ">

            <div style="
            width:{confidence*100:.2f}%;
            height:100%;
            background:linear-gradient(90deg,#06b6d4,#3b82f6);
            border-radius:30px;
            transition:1s;
            "></div>

            </div>

            </div>
            """, unsafe_allow_html=True)


else:

    with right:

        st.markdown("""
        <div class="card">

        <h2 style="text-align:center;">
        🤖 Ready to Predict
        </h2>

        <br>

        <p style="text-align:center;color:#94a3b8;font-size:18px;">

        Upload a handwritten English alphabet image.

        Supported Formats

        ✔ PNG

        ✔ JPG

        ✔ JPEG

        </p>

        </div>

        """, unsafe_allow_html=True)

# ==================================================
# Model Information
# ==================================================

st.markdown(
"""

<div style="height:2px;
background:rgba(255,255,255,.08);
margin:35px 0;">
</div>
""",
unsafe_allow_html=True
)

st.markdown("""
<h2 style="margin-bottom:20px;">

📊 Model Overview

</h2>

<p style="color:#94a3b8;">

Model performance and architecture summary.

</p>

""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown("""
    <div class="metric">

    <h2>94.81%</h2>

    <p>Accuracy</p>

    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown("""
    <div class="metric">

    <h2>26</h2>

    <p>Classes</p>

    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown("""
    <div class="metric">

    <h2>CNN</h2>

    <p>Architecture</p>

    </div>
    """, unsafe_allow_html=True)

with c4:

    st.markdown("""
    <div class="metric">

    <h2>EMNIST</h2>

    <p>Dataset</p>

    </div>
    """, unsafe_allow_html=True)



# ==================================================
# Footer
# ==================================================

st.divider()

_, center, _ = st.columns([1, 3, 1])

with center:

    st.markdown(
        "<h4 style='text-align:center;'>✍️ Handwritten Character Recognition</h4>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h6 style='text-align:center;'>Developed using TensorFlow • CNN • EMNIST Letters • Streamlit</h6>",unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;color:gray;'>© 2026 | Machine Learning Internship Project</p>",
        unsafe_allow_html=True
    )