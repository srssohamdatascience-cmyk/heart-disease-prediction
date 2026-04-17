import os
import pickle
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'heart_model.pkl')

try:
    model = pickle.load(open(model_path, 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")
    
st.title("❤️ Heart Disease Prediction")
st.markdown('Enter Patient details to predict risk')

age = st.slider("Age", 10, 100)

sex = st.selectbox('Gender', ['M', 'F'])

chestpaintype = st.selectbox('Chest Pain Type', ['TA','ATA','NAP','ASY'])

restingbp = st.number_input('Resting Blood Pressure', value=120)

cholesterol = st.number_input('Cholesterol', value=200)

fastingbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

restingecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

exerciseangina = st.selectbox("Exercise Induced Angina", ["Y", "N"])

maxhr = st.number_input("Max Heart Rate", value=150)

oldpeak = st.number_input("Oldpeak", value=1.0)

st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])



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


if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease")
        st.write(f"Risk Probability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk of Heart Disease")
        st.write(f"Risk Probability: {probability:.2%}")
        
        
    if probability < 0.3:
      st.info("🟢 Low Risk")
    elif probability < 0.7:
        st.warning("🟡 Moderate Risk")
    else:
       st.error("🔴 High Risk")        