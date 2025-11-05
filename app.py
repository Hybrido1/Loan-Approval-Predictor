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
st.title("Loan Approval Predictor App")

st.markdown("""
This app predicts the **probability of your loan getting approved** based on your details.
Please enter the following information:
""")

# --- User Inputs based on actual dataset ---
no_of_dependents = st.number_input("Number of dependents", min_value=0, max_value=10, step=1)
income_annum = st.number_input("Income Annum", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term", min_value=2, max_value=20, step=1)
cibil_score = st.number_input("Cibil Score", min_value =300, max_value = 900, step = 1 )
residential_assets_value = st.number_input("Residential Assets Value", min_value = 0)
commercial_assets_value  = st.number_input("Commercial Assets Value", min_value = 0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value = 0)
bank_asset_value = st.number_input("Bank Asset Value", min_value  = 0)
self_employed = st.selectbox("Self_employed", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Non-Graduate"])

# --- Create dataframe for model ---
input_data =pd.DataFrame({
    'Number of dependents': [no_of_dependents],
    'Income Annum': [income_annum],
    'Loan Amount': [loan_amount],
    'Loan Term': [loan_term],
    'Cibil Score': [cibil_score],
    'Residential Assets Value': [residential_assets_value],
    'Commercial Assets Value': [commercial_assets_value],
    'Luxury Assets Value' : [luxury_assets_value],
    'Bank Asset Value' : [bank_asset_value],
    'Self Employed' : [self_employed],
    'Education' : [education]
})

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





