# ===============================
# loan_prediction_app.py
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# -------------------------------
# 1️⃣ Title
# -------------------------------
st.title("🏦 Loan Approval Prediction App")

# -------------------------------
# 2️⃣ Upload CSV file
# -------------------------------
uploaded_file = st.file_uploader("Upload your Loan Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # 3️⃣ EDA Section
    # -------------------------------
    st.subheader("🔍 Exploratory Data Analysis")

    st.write("**Basic Info:**")
    st.write(df.describe())

    # Correlation heatmap
    st.write("**Correlation Matrix:**")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Distribution plot for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_to_plot = st.selectbox("Select a feature to view distribution:", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[feature_to_plot], kde=True, color='teal', ax=ax)
    st.pyplot(fig)

    # -------------------------------
    # 4️⃣ Data Preprocessing
    # -------------------------------
    st.subheader("⚙️ Data Preprocessing")

    # Example categorical mappings
    mappings = {
        "Married": {"Yes": 1, "No": 0},
        "Education": {"Graduate": 1, "Non-Graduate": 0},
        "Gender": {"Male": 1, "Female": 0}
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    st.write("✅ Applied One-Hot Encoding via Mapping")

    # Drop rows with missing values
    df.dropna(inplace=True)

    # -------------------------------
    # 5️⃣ Model Training
    # -------------------------------
    st.subheader("🤖 Model Training")

    target_col = st.selectbox("Select Target Column (Loan Approval)", df.columns)
    features = [col for col in df.columns if col != target_col]

    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("🎯 Model Accuracy", f"{acc*100:.2f}%")

    # -------------------------------
    # 6️⃣ Confusion Matrix
    # -------------------------------
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap='Greens')
    st.pyplot(fig)

    # -------------------------------
    # 7️⃣ Make Predictions
    # -------------------------------
    st.subheader("🧩 Try Prediction Manually")

    input_data = {}
    for feature in features:
        if df[feature].dtype == 'object':
            input_data[feature] = st.selectbox(f"{feature}:", df[feature].unique())
        else:
            input_data[feature] = st.number_input(f"{feature}:", float(df[feature].min()), float(df[feature].max()))

    if st.button("Predict Loan Approval"):
        input_df = pd.DataFrame([input_data])
        # Apply same mapping and scaling
        for col, mapping in mappings.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].map(mapping)
        input_df = input_df.fillna(0)
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)
        result = "✅ Loan Approved" if pred[0] == 1 else "❌ Loan Rejected"
        st.success(result)

else:
    st.info("👆 Please upload a dataset to begin.")
