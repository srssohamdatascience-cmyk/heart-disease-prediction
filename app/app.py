import os
import pickle
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ================== LOAD MODEL ==================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'heart_model.pkl')

try:
    model = pickle.load(open(model_path, 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# ================== SIDEBAR ==================
with st.sidebar:
    st.title("🧠 About")
    st.write("This app predicts heart disease risk using Machine Learning.")
    st.write("Fill the patient details to get instant prediction.")
    if st.button("🔄 Reset"):
        st.rerun()

# ================== TITLE ==================
st.title("❤️ Heart Disease Prediction")
st.markdown("### Smart risk analysis dashboard")

# ================== MAPPINGS ==================
chest_pain_map = {
    "Typical Angina (Pain during activity)": "TA",
    "Atypical Angina (Unusual pain)": "ATA",
    "Non-Anginal Pain": "NAP",
    "No Symptoms": "ASY"
}

restingecg_map = {
    "Normal": "Normal",
    "ST Abnormality": "ST",
    "Left Ventricular Hypertrophy": "LVH"
}

# ================== INPUT UI ==================
col1, col2 = st.columns(2)

with col1:
    with st.expander("🧾 Basic Information", expanded=True):
        age = st.number_input("Age", 10, 100, 25)
        sex = st.radio("Gender", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", list(chest_pain_map.keys()))

    with st.expander("❤️ Heart Metrics", expanded=True):
        restingbp = st.number_input("Resting Blood Pressure", value=120)
        cholesterol = st.number_input("Cholesterol", value=200)
        maxhr = st.number_input("Max Heart Rate", value=150)

with col2:
    with st.expander("📊 Clinical Indicators", expanded=True):
        fastingbs = st.radio("Fasting Blood Sugar > 120?", ["Yes", "No"])
        restingecg = st.selectbox("ECG Result", list(restingecg_map.keys()))
        exerciseangina = st.radio("Exercise Angina?", ["Yes", "No"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", value=1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ================== CONVERT INPUT ==================
sex = "M" if sex == "Male" else "F"
fastingbs = 1 if fastingbs == "Yes" else 0
exerciseangina = "Y" if exerciseangina == "Yes" else "N"
chestpaintype = chest_pain_map[chest_pain]
restingecg = restingecg_map[restingecg]

# ================== DATAFRAME ==================
input_df = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'chestpaintype': [chestpaintype],
    'restingbp': [restingbp],
    'cholesterol': [cholesterol],
    'fastingbs': [fastingbs],
    'restingecg': [restingecg],
    'exerciseangina': [exerciseangina],
    'maxhr': [maxhr],
    'oldpeak': [oldpeak],
    'st_slope': [st_slope]
})

# ================== PREDICTION (REAL-TIME) ==================
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# ================== HERO SECTION ==================
st.markdown("## 📊 Risk Analysis")

colA, colB = st.columns([2, 1])

with colA:
    st.metric("Risk Probability", f"{probability:.2%}")

with colB:
    if probability < 0.3:
        st.success("🟢 Low Risk")
    elif probability < 0.7:
        st.warning("🟡 Moderate Risk")
    else:
        st.error("🔴 High Risk")

# ================== GAUGE CHART ==================
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability * 100,
    title={'text': "Heart Risk (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'steps': [
            {'range': [0, 30], 'color': "green"},
            {'range': [30, 70], 'color': "yellow"},
            {'range': [70, 100], 'color': "red"},
        ],
    }
))

st.plotly_chart(fig, use_container_width=True)

# ================== SMART FEEDBACK ==================
st.markdown("## 💡 Health Insights")

if cholesterol > 240:
    st.warning("⚠️ High cholesterol detected")

if restingbp > 140:
    st.warning("⚠️ High blood pressure")

if maxhr < 100:
    st.info("ℹ️ Low heart rate capacity")

# ================== FINAL RESULT ==================
st.markdown("## 🧾 Final Prediction")

if prediction == 1:
    st.error("⚠️ High Risk of Heart Disease")
else:
    st.success("✅ Low Risk of Heart Disease")

# ================== DISCLAIMER ==================
st.caption("⚠️ This tool is for educational purposes only.")