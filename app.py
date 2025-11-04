import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
try:
    model = joblib.load("loan_prediction_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Loan Approval Predictor App")

st.markdown("""
This app predicts the **probability of your loan getting approved** based on your details.
Please enter the following information:
""")

# --- User Inputs based on actual dataset ---
age = st.number_input("Age", min_value=18, max_value=100, step=1)
income = st.number_input("Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Non-Graduate"])
gender = st.selectbox("Gender", ["Male", "Female"])

# --- Create dataframe for model ---
input_dict = {
    'Age': age,
    'Income': income,
    'LoanAmount': loan_amount,
    'CreditScore': credit_score,
    'Married': married,
    'Education': education,
    'Gender': gender
}

input_df = pd.DataFrame([input_dict])

# --- Prediction Section ---
if st.button("Predict Loan Approval"):
    try:
        # Model predicts probability
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
        st.error(f"Error during prediction: {e}")
