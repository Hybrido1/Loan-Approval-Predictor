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
Enter all required fields below and click **Predict Loan Approval**.
""")

# --- User Inputs ---
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
income_annum = st.number_input("Annual Income (₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, step=1)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0)
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Non-Graduate"])

# --- Create DataFrame (must match model columns) ---
input_df = pd.DataFrame({
    'no_of_dependents': [no_of_dependents],
    'income_annum': [income_annum],
    'loan_amount': [loan_amount],
    'loan_term': [loan_term],
    'cibil_score': [cibil_score],
    'residential_assets_value': [residential_assets_value],
    'commercial_assets_value': [commercial_assets_value],
    'luxury_assets_value': [luxury_assets_value],
    'bank_asset_value': [bank_asset_value],
    'self_employed': [self_employed],
    'education': [education]
})

# --- Prediction Section ---
if st.button("Predict Loan Approval"):
    try:
        # Probability prediction
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1] * 100
        else:
            prediction = model.predict(input_df)[0]
            probability = prediction * 100 if 0 <= prediction <= 1 else np.clip(prediction, 0, 100)

        # Display results
        if probability > 55:
            st.success(f"✅ Loan Approved with {probability:.2f}% confidence.")
        else:
            st.error(f"❌ Loan Not Approved ({probability:.2f}% probability).")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
