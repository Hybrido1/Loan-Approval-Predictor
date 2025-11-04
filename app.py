import streamlit as st
import pickle
import pandas as pd

# Load trained model

with open("loan_prediction_model.pkl", "rb") as file:
model = pickle.load(file)

st.title("🏦 Loan Approval Predictor App")

st.markdown("""
This app predicts the **probability of your loan getting approved** based on your details.
Enter the required information below:
""")

# --- Example Inputs (You can change names/labels based on your dataset) ---

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# --- Create input dataframe (must match training order) ---

input_dict = {
'Gender': gender,
'Married': married,
'Dependents': dependents,
'Education': education,
'Self_Employed': self_employed,
'ApplicantIncome': applicant_income,
'CoapplicantIncome': coapplicant_income,
'LoanAmount': loan_amount,
'Loan_Amount_Term': loan_term,
'Credit_History': credit_history,
'Property_Area': property_area
}

input_data = pd.DataFrame([input_dict])

# --- Predict Section ---

if st.button("Predict Loan Approval"):
try:
prediction = model.predict(input_data)[0]

```
    # Convert to % if value is between 0 and 1
    probability = prediction * 100 if 0 <= prediction <= 1 else prediction

    if probability > 55:
        st.success(f"✅ Loan Approved with {probability:.2f}% confidence.")
    else:
        st.error(f"❌ Loan Not Approved ({probability:.2f}% probability).")

except Exception as e:
    st.error(f"Error while predicting: {e}")
```

