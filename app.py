
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CalorieIQ | Fitness Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "calories_model.joblib"

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #090d16;
    --panel: #111827;
    --muted: #94a3b8;
    --accent: #a3ff5f;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #183125 0%, transparent 35%),
        radial-gradient(ellipse at 100% 20%, #172347 0%, transparent 30%),
        var(--bg);
    color: #f8fafc;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.7px;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #111827;
}

.hero {
    padding: 34px 0 28px;
}

.eyebrow {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 12px;
    font-weight: 700;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(38px, 6vw, 66px);
    font-weight: 700;
    line-height: 1.06;
    margin: 14px 0;
}

.hero-title span {
    color: var(--accent);
}

.hero-sub {
    color: #a7b2c5;
    font-size: 16px;
    max-width: 600px;
    line-height: 1.8;
}

.panel {
    background: rgba(17, 24, 39, 0.88);
    border: 1px solid #283448;
    border-radius: 22px;
    padding: 24px;
    margin-bottom: 18px;
}

.result-card {
    background: linear-gradient(135deg, #c9ff8b 0%, #8ff0a4 55%, #6ee7d8 100%);
    color: #0b1711;
    border-radius: 22px;
    padding: 30px;
    min-height: 220px;
    margin-bottom: 18px;
}

.result-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
}

.result-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(44px, 6vw, 68px);
    font-weight: 700;
    line-height: 1.2;
    margin: 16px 0 8px;
}

.result-note {
    font-size: 13px;
    opacity: 0.8;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #0c1320;
    border-radius: 10px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: var(--accent);
    color: #10180d;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    min-height: 48px;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #c0ff91;
    color: #10180d;
    transform: translateY(-1px);
}

div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #283448;
    padding: 18px;
    border-radius: 16px;
}

div[data-testid="stMetricLabel"] {
    color: #a7b2c5;
}

.footer {
    text-align: center;
    color: #718096;
    font-size: 12px;
    padding-top: 25px;
}

hr {
    border-color: #283448;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


bundle = load_model()

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">⚡ Fitness intelligence platform</div>
    <div class="hero-title">
        Train smarter.<br>
        <span>Know your burn.</span>
    </div>
    <div class="hero-sub">
        Estimate calories burned during your workout using a
        machine-learning model and your personal workout details.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚡ CalorieIQ")
    st.caption("Your workout prediction companion")
    st.markdown("---")
    st.markdown("### How it works")
    st.write("1. Enter your personal details.")
    st.write("2. Add your workout measurements.")
    st.write("3. Generate your calorie estimate.")
    st.markdown("---")
    st.caption("Powered by a Random Forest regression model.")


if bundle is None:
    st.error(
        "The trained model was not found. Run `python train_model.py` "
        "in your project terminal first."
    )
    st.stop()

model = bundle["model"]
features = bundle["features"]

# ---------------- INPUTS ----------------
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown("## Your workout profile")
    st.caption("Enter the measurements for the workout you want to estimate.")

    with st.form("prediction_form"):
        st.markdown("### Personal details")

        gender = st.selectbox("Gender", ["Male", "Female"])

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input(
                "Age (years)", min_value=1, max_value=100, value=25
            )
            height = st.number_input(
                "Height (cm)", min_value=50.0, max_value=250.0,
                value=170.0, step=1.0
            )

        with c2:
            weight = st.number_input(
                "Weight (kg)", min_value=10.0, max_value=350.0,
                value=70.0, step=0.5
            )
            duration = st.number_input(
                "Workout duration (minutes)",
                min_value=1.0, max_value=300.0,
                value=30.0, step=1.0
            )

        st.markdown("### Workout measurements")

        c3, c4 = st.columns(2)
        with c3:
            heart_rate = st.number_input(
                "Heart rate (BPM)", min_value=30, max_value=240,
                value=100
            )

        with c4:
            body_temp = st.number_input(
                "Body temperature (°C)", min_value=25.0,
                max_value=45.0, value=37.0, step=0.1,
                format="%.1f"
            )

        submitted = st.form_submit_button(
            "⚡  Predict Calories Burnt",
            use_container_width=True
        )

with right:
    st.markdown("## Your results")

    if "calorie_prediction" not in st.session_state:
        st.markdown("""
        <div class="result-card">
            <div class="result-label">Estimated energy burn</div>
            <div class="result-value">-- <span style="font-size:22px">kcal</span></div>
            <div class="result-note">
                Your prediction will appear here after you submit your details.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
            <h3>Ready when you are.</h3>
            <p style="color:#a7b2c5;line-height:1.8">
                Complete the workout profile and select Predict Calories Burnt
                to generate your estimate.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        prediction = st.session_state["calorie_prediction"]
        details = st.session_state["workout_details"]

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated calories burned</div>
            <div class="result-value">{prediction:,.1f}
                <span style="font-size:22px">kcal</span>
            </div>
            <div class="result-note">
                Machine-learning estimate based on your submitted measurements.
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Workout duration", f"{details['duration']:.0f} min")
        with m2:
            st.metric("Heart rate", f"{details['heart_rate']} BPM")

        st.markdown("""
        <div class="panel">
            <h3>Workout summary</h3>
            <p style="color:#a7b2c5">
                Your estimate is generated from the trained model using
                your age, body measurements, workout duration, heart rate,
                body temperature, and gender.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- PREDICT ----------------
if submitted:
    gender_value = 0 if gender == "Male" else 1

    input_values = {
        "Gender": gender_value,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
    }

    input_df = pd.DataFrame(
        [[input_values[col] for col in features]],
        columns=features,
    )

    try:
        result = float(model.predict(input_df)[0])

        if not pd.notna(result):
            st.error("The model returned an invalid prediction.")
        else:
            st.session_state["calorie_prediction"] = max(0.0, result)
            st.session_state["workout_details"] = {
                "duration": duration,
                "heart_rate": heart_rate,
            }
            st.rerun()

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.markdown("---")
st.markdown("""
<div class="footer">
    CALORIEIQ · MACHINE LEARNING PROJECT<br><br>
    Predictions are estimates for educational purposes and are not
    medical measurements or a substitute for professional advice.
</div>
""", unsafe_allow_html=True)
