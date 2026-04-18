import os
import pickle
import streamlit as st
import pandas as pd

# ================== LOAD MODEL ==================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'heart_model.pkl')

try:
    model = pickle.load(open(model_path, 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction")
st.markdown("### Enter patient details to assess risk")

# ================== MAPPINGS ==================
chest_pain_map = {
    "Typical Angina (Chest pain during activity)": "TA",
    "Atypical Angina (Unusual chest pain)": "ATA",
    "Non-Anginal Pain (Not heart-related)": "NAP",
    "No Symptoms (Asymptomatic)": "ASY"
}

# ================== INPUT UI ==================
st.header("🧾 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25,
                          help="Patient's age in years")

    sex = st.radio("Gender", ["Male", "Female"])

    chest_pain = st.selectbox("Chest Pain Type", list(chest_pain_map.keys()))

    restingbp = st.number_input("Resting Blood Pressure", value=120,
                                help="Normal range: ~90–120 mm Hg")

    cholesterol = st.number_input("Cholesterol Level", value=200,
                                  help="Measured in mg/dL. Ideal < 200")

with col2:
    fastingbs = st.radio("Fasting Blood Sugar > 120 mg/dL?", ["Yes", "No"])

    restingecg = st.selectbox("Resting ECG Result", [
        "Normal",
        "ST Abnormality",
        "Left Ventricular Hypertrophy"
    ])

    exerciseangina = st.radio("Exercise-induced chest pain?", ["Yes", "No"])

    maxhr = st.number_input("Maximum Heart Rate Achieved", value=150)

    oldpeak = st.number_input("ST Depression (Oldpeak)", value=1.0,
                              help="Lower is generally better")

    st_slope = st.selectbox("ST Segment Slope", ["Up", "Flat", "Down"])

# ================== CONVERT INPUT ==================
sex = "M" if sex == "Male" else "F"
fastingbs = 1 if fastingbs == "Yes" else 0
exerciseangina = "Y" if exerciseangina == "Yes" else "N"
chestpaintype = chest_pain_map[chest_pain]

# ECG mapping (match training data!)
restingecg_map = {
    "Normal": "Normal",
    "ST Abnormality": "ST",
    "Left Ventricular Hypertrophy": "LVH"
}
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

# ================== PREDICTION ==================
if st.button("🔍 Predict Risk"):

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("📊 Prediction Result")

        # Metric display
        st.metric("Risk Probability", f"{probability:.2%}")

        # Progress bar
        st.progress(float(probability))

        # Risk interpretation
        if probability < 0.3:
            st.success("🟢 Low Risk: Maintain a healthy lifestyle.")
        elif probability < 0.7:
            st.warning("🟡 Moderate Risk: Consider consulting a doctor.")
        else:
            st.error("🔴 High Risk: Medical consultation recommended.")

        # Final classification
        if prediction == 1:
            st.error("⚠️ Model Prediction: High Risk of Heart Disease")
        else:
            st.success("✅ Model Prediction: Low Risk of Heart Disease")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ================== DISCLAIMER ==================
st.caption("⚠️ This tool is for educational purposes only and not a medical diagnosis.")