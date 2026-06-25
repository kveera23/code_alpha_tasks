# ==========================================================
# IMPORTS
# ==========================================================
import os
import tempfile
import streamlit as st

from src.predict import predict_emotion


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)


# ==========================================================
# EMOJI DICTIONARY
# ==========================================================
emoji = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Fear": "😨",
    "Disgust": "🤢",
    "Surprise": "😲",
    "Neutral": "😐",
    "Calm": "😌"
}


# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

header{visibility:hidden;}
footer{visibility:hidden;}
#MainMenu{visibility:hidden;}

/* Background */

.stApp{

background:
linear-gradient(
180deg,
#F8FAFC 0%,
#F0FDFA 45%,
#ECFDF5 100%
);

}



.stProgress > div > div > div{

background:#10B981;

border-radius:20px;

}

.stProgress{

margin-top:8px;

margin-bottom:18px;

}
            
                     
.prediction-card{

background:linear-gradient(135deg,#0F766E,#14B8A6);

border-radius:25px;

padding:35px;

text-align:center;

color:white;

box-shadow:0 15px 35px rgba(0,0,0,.15);

}

.result-title{

font-size:30px;

font-weight:700;

margin-bottom:20px;

}

.result-emoji{

font-size:95px;

margin-bottom:10px;

}

.result-emotion{

font-size:48px;

font-weight:700;

margin-bottom:20px;

}

.result-label{

font-size:22px;

opacity:.95;

}

.result-confidence{

font-size:42px;

font-weight:700;

margin-top:10px;

}

.result-status{

margin-top:20px;

font-size:18px;

opacity:.95;

}




/* Page */

.block-container{

padding-top:2rem;
padding-left:5rem;
padding-right:5rem;
padding-bottom:0rem;

}


/* Hero */

.hero{

padding:50px;

border-radius:28px;

background: linear-gradient(
135deg,
#0F766E 0%,
#0D9488 45%,
#14B8A6 100%
);

color:white;

text-align:center;

box-shadow:0 18px 40px rgba(0,0,0,.15);

margin-bottom:40px;

}

.hero h1{

font-size:56px;

font-weight:700;

margin-bottom:10px;

}

.hero p{

font-size:22px;

opacity:.95;

}


/* Cards */

.card{

background:white;

padding:28px;

border-radius:22px;

box-shadow:
0 12px 35px rgba(15,118,110,.10);

margin-bottom:20px;

transition:.35s;

}

.card:hover{

transform:translateY(-5px);

}

.card-title{

font-size:24px;

font-weight:600;

margin-bottom:15px;

color:#1E293B;

}


/* Upload */

[data-testid="stFileUploader"]{

border:2px dashed #14B8A6;

border-radius:18px;

padding:18px;

}


/* Audio */

audio{

width:100%;

}


/* Button */

.stButton>button{

height:65px;

width:100%;

border:none;

border-radius:18px;

font-size:24px;

font-weight:600;

color:white;

background:
linear-gradient(
135deg,
#0F766E,
#0D9488
);

transition:.35s;

}

.stButton>button:hover{

transform:scale(1.02);

box-shadow:0 10px 30px rgba(16,185,129,.35);
background:
linear-gradient(
135deg,
#115E59,
#0F766E
);

}



.stInfo{

border-radius:18px;

background:#F8FAFC;

border:none;

}

.stSuccess{

border-radius:18px;

}

/* ======================================
   RESULT CARD
====================================== */

.result-card{

background:linear-gradient(135deg,#0F766E,#14B8A6);

border-radius:25px;

padding:35px;

text-align:center;

color:white;

box-shadow:0 15px 35px rgba(20,184,166,.25);

height:100%;

}

.result-emoji{

font-size:90px;

}

.result-emotion{

font-size:46px;

font-weight:700;

margin-top:10px;

}

.result-confidence{

font-size:34px;

font-weight:600;

margin-top:15px;

}


/* ======================================
   INFO CARD
====================================== */

.info-card{

background:white;

border-radius:20px;

padding:20px;

box-shadow:0 10px 25px rgba(0,0,0,.08);

text-align:center;

margin-bottom:18px;

height:130px;

}

.info-title{

font-size:16px;

color:#64748B;

}

.info-value{

font-size:28px;

font-weight:700;

color:#0F766E;

margin-top:12px;

}


/* ======================================
   SCORE TITLE
====================================== */

.score-title{

font-size:30px;

font-weight:700;

margin-top:35px;

margin-bottom:20px;

color:#0F172A;

}            

/* Metrics */

[data-testid="stMetric"]{

background:white;

padding:18px;

border-radius:18px;

box-shadow:0 10px 25px rgba(0,0,0,.08);

border:none;

}

[data-testid="stMetricValue"]{

font-size:32px;

color:#0F766E;

font-weight:700;

}

[data-testid="stMetricLabel"]{

font-size:16px;

font-weight:600;

}

/* Progress */

.stProgress > div > div > div{

background:#10B981;

border-radius:10px;

}

.stProgress{

margin-top:8px;

margin-bottom:20px;

}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# HERO SECTION
# ==========================================================
st.markdown("""

<div class="hero">

<h1>🎤 Emotion Recognition From Speech</h1>

<p>
Analyze Human Emotions Using Deep Learning (CNN + BiLSTM)
</p>

</div>

""", unsafe_allow_html=True)


# ==========================================================
# AUDIO UPLOAD
# ==========================================================
uploaded_file = st.file_uploader(
    "Upload a WAV Audio File",
    type=["wav"]
)


# ==========================================================
# AUDIO PREVIEW SECTION
# ==========================================================
if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="card">
        <div class="card-title">
        📤 Upload Status
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="
            background:linear-gradient(135deg,#0F766E,#10B981);
            color:white;
            padding:14px;
            border-radius:12px;
            text-align:center;
            font-size:18px;
            font-weight:600;
            ">
            ✅ Speech Uploaded Successfully!
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">
        <div class="card-title">
        🎵 Audio Preview
        </div>
        """, unsafe_allow_html=True)

        st.audio(uploaded_file)

        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# ANALYZE BUTTON
# ==========================================================
if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:

        tmp.write(uploaded_file.read())

        temp_audio_path = tmp.name

    left, center, right = st.columns([1, 2, 1])

    with center:

        predict = st.button(
            "🧠 Analyze Emotion",
            use_container_width=True
        )

    if predict:

        with st.spinner("Analyzing Speech..."):

            emotion, confidence, probabilities = predict_emotion(
                temp_audio_path
            )

        st.toast("Prediction Completed Successfully ✅")

    
        # ==========================================================
        # PREDICTION DASHBOARD
        # ==========================================================

        st.markdown("""
            <h2 style="
            color:#0F766E;
            font-weight:700;
            margin-bottom:15px;">
            🎯 Prediction Dashboard
            </h2>
            """, unsafe_allow_html=True)

        left, right = st.columns([1.3, 1])

        # =============================
        # LEFT PANEL
        # =============================

        with left:

            st.markdown(f"""
            <div class="prediction-card">

            <div class="result-title">
            🎯 Prediction Result
            </div>

            <div class="result-emoji">
            {emoji.get(emotion,"🎤")}
            </div>

            <div class="result-emotion">
            {emotion}
            </div>

            <div class="result-label">
            Prediction Confidence
            </div>

            <div class="result-confidence">
            {confidence*100:.2f}%
            </div>

            <div class="result-status">
            ✔ Prediction Completed Successfully
            </div>

            </div>
            """, unsafe_allow_html=True)

        # =============================
        # RIGHT PANEL
        # =============================

        with right:

            c1, c2 = st.columns(2)

            c1.metric(
                "📂 Dataset",
                "RAVDESS"
            )

            c2.metric(
                "🧠 Model",
                "CNN + BiLSTM"
            )

            c3, c4 = st.columns(2)

            c3.metric(
                "🎯 Accuracy",
                "85.76%"
            )

            c4.metric(
                "🎵 Features",
                "172"
            )

        st.markdown("---")

        st.markdown("""
                <h2 style="
                color:#0F766E;
                font-weight:700;
                margin-bottom:15px;">
                📊 Emotion Confidence Scores
                </h2>
                """, unsafe_allow_html=True)

        # Sort probabilities
        sorted_probs = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for emo, score in sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            c1, c2, c3 = st.columns([1, 8, 0.5])

            with c1:
                st.markdown(
                    f"""
                    <div style="
                        color:#0F172A;
                        font-size:20px;
                        font-weight:600;
                    ">
                        {emoji.get(emo,'🎭')} {emo}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:
                st.progress(float(score))

            with c3:
                st.markdown(
                    f"<div style='text-align:right;font-weight:600;color:#0F766E;'>{score*100:.2f}%</div>",
                    unsafe_allow_html=True
                )

        try:
            os.remove(temp_audio_path)
        except:
            pass

    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)



# ==========================================================
# FOOTER
# ==========================================================


st.markdown("""
<div style="
text-align:center;
color:#6B7280;
font-size:15px;
padding:4px 0px;
margin-bottom:0px;
margin-top:15px;
">

Speech Emotion Recognition using CNN + BiLSTM<br>
Developed using TensorFlow • Streamlit • Librosa

</div>
""", unsafe_allow_html=True)