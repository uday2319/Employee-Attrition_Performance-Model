import streamlit as st
import pandas as pd
import joblib

st.title("Employee Attrition Prediction App")

pipeline = joblib.load("employee_attrition.pkl")

age = st.number_input("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 30000, 5000)
gender = st.selectbox("Gender", ["Male", "Female"])
job_role = st.selectbox("Job Role", ["Research Scientist", "Sales Executive", "Manager"])

data = {
    "Age": age,
    "Gender": gender,
    "MonthlyIncome": monthly_income,
    "JobRole": job_role
}

df = pd.DataFrame([data])

if st.button("Predict"):
    pred = pipeline.predict(df)[0]
    st.write("Attrition: **Yes**" if pred == 1 else "Attrition: **No**")
