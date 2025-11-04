import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model pipeline
model = joblib.load("loan_prediction_model.pkl")

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")

st.title("💰 Loan Approval Prediction App")
st.write("Enter your details below to predict your loan approval chance.")

# --- Collect Inputs ---
st.header("Applicant Information")

# Numerical Inputs
age = st.number_input("Age", min_value=18, max_value=75, value=30)
income = st.number_input("Monthly Income ($)", min_value=1000, max_value=100000, value=5000)
loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=50000, value=10000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

# Categorical Inputs
married = st.selectbox("Marital Status", ["Yes", "No"])
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
gender = st.selectbox("Gender", ["Male", "Female"])

# --- Create DataFrame for Model ---
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "LoanAmount": [loan_amount],
    "CreditScore": [credit_score],
    "Married": [married],
    "Education": [education],
    "Gender": [gender]
})

# --- Prediction ---
if st.button("🔍 Predict Loan Approval"):
    probability = model.predict_proba(input_data)[0][1] * 100  # Probability of being Approved
    st.write(f"### Prediction Confidence: {probability:.2f}%")

    if probability >= 55:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Not Approved.")
