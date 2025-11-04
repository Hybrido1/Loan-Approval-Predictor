import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load trained model
with open("loan_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Loan Approval Predictor App")

st.markdown("""
This app predicts the **probability of your loan getting approved** based on your details.
Enter the required information below:
""")

# --- Example Inputs ---
age = st.number_input("Age", min_val = 18)
applicant_income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
Credit_score = st.number_input("Credit Score")
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
gender = st.selectbox("Gender", ["Male", "Female"])
# --- Create input dataframe (must match training order) ---
input_dict = {
    'Age' : Age,
    'ApplicantIncome': applicant_income,
    'LoanAmount': loan_amount,
    'Credit Score' : Credit_score,
    'Married': married,
    'Education': education,
    'Gender': gender
}

input_df = pd.DataFrame([input_dict])

# --- Predict Section ---
if st.button("Predict Loan Approval"):
    try:
        # If model supports probabilities
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1] * 100
        else:
            prediction = model.predict(input_df)[0]
            probability = prediction * 100 if 0 <= prediction <= 1 else np.clip(prediction, 0, 100)

        if probability > 55:
            st.success(f"✅ Loan Approved with {probability:.2f}% confidence.")
        else:
            st.error(f"❌ Loan Not Approved ({probability:.2f}% probability).")

    except Exception as e:
        st.error(f"Error while predicting: {e}")

st.markdown("---")

