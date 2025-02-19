import streamlit as st
import numpy as np
import joblib  

model = joblib.load("xgb_model.pkl")  
scaler = joblib.load("scaler.pkl")    

feature_names = [
    "Age", "Gender", "BMI", "AlcoholConsumption", "Smoking", "GeneticRisk",
    "PhysicalActivity", "Diabetes", "Hypertension", "LiverFunctionTest"
]

st.set_page_config(page_title="Liver Disease Prediction", layout="wide")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4762/4762313.png", width=150)
st.sidebar.title("🔬 About the Model")
st.sidebar.info("This AI model predicts liver disease based on patient details using an XGBoost classifier.")

st.title("🏥 Liver Disease Prediction System")
st.write("Enter the patient details below to predict the likelihood of liver disease.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🧑 Age", min_value=1, max_value=100, step=1)
    bmi = st.number_input("⚖️ BMI (Body Mass Index)", min_value=10.0, max_value=50.0, step=0.1)
    alcohol = st.slider("🍺 Alcohol Consumption (per week)", min_value=0, max_value=20, step=1)
    smoking = st.selectbox("🚬 Smoking Habit", ["No", "Yes"])
    physical_activity = st.slider("🏃 Physical Activity (hours per week)", min_value=0, max_value=20, step=1)

with col2:
    gender = st.selectbox("⚤ Gender", ["Male", "Female"])
    genetic_risk = st.selectbox("🧬 Genetic Risk Factor", ["Low", "Medium", "High"])
    diabetes = st.selectbox("🩸 Diabetes", ["No", "Yes"])
    hypertension = st.selectbox("💓 Hypertension", ["No", "Yes"])
    liver_test = st.slider("🧪 Liver Function Test Score", min_value=0.0, max_value=100.0, step=0.1)

gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
hypertension = 1 if hypertension == "Yes" else 0
genetic_risk = {"Low": 0, "Medium": 1, "High": 2}[genetic_risk]

input_data = np.array([[age, gender, bmi, alcohol, smoking, genetic_risk, physical_activity, diabetes, hypertension, liver_test]])
scaled_data = scaler.transform(input_data) 
if st.button("🔍 Predict"):
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1] 

    st.subheader("🩺 Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ The patient is **likely** to have liver disease. (Risk Score: {probability:.2%})")
    else:
        st.success(f"✅ The patient is **not likely** to have liver disease. (Risk Score: {probability:.2%})")


    st.info("This prediction is based on medical data and should be reviewed by a healthcare professional.")

st.markdown("""
---
👨‍⚕️ **Disclaimer:** This AI model provides **only a prediction** and does not replace professional medical diagnosis.
""")
