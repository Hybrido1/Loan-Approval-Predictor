#  Loan Approval Predictor App

A **Machine Learning–based web application** built with **Streamlit** that predicts the probability of a loan getting approved based on user inputs such as income, CIBIL score, assets, and loan amount.

---

##  Overview

This project demonstrates an end-to-end **Loan Approval Prediction System** using **Logistic Regression**.
It includes data cleaning, feature encoding, exploratory data analysis (EDA), model training, evaluation, and deployment using **Streamlit**.

The application takes user inputs and returns:

* The **probability (in %)** that the loan will be approved
* A **decision message** (`Loan Approved` or `Loan Not Approved`)

---

##  Model Summary

* **Algorithm Used:** Logistic Regression
* **Accuracy Achieved:** ~85–90% (after preprocessing and scaling)
* **Training Dataset:** Synthetic loan dataset with features like:

  * Applicant income
  * Loan amount
  * Loan term
  * CIBIL score
  * Asset values (Residential, Commercial, Luxury, Bank)
  * Education level
  * Employment type
  * Number of dependents

---

##  Tech Stack

| Component            | Technology          |
| -------------------- | ------------------- |
| Programming Language | Python 3.9+         |
| Framework            | Streamlit           |
| ML Library           | scikit-learn        |
| Data Processing      | pandas, numpy       |
| Visualization        | matplotlib, seaborn |
| Model Serialization  | joblib / pickle     |

---

##  Project Structure

```
📦 Loan-Approval-Predictor
├── app.py                       # Streamlit app
├── loan_prediction_model.pkl    # Trained Logistic Regression model
├── bank_loan_data.csv           # Processed training data
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── notebooks/
    └── Loan_Approval_Training.ipynb   # Model training and EDA notebook
```

---

##  Features

* Clean, minimal, and interactive UI built with Streamlit
* Real-time probability calculation of loan approval
* Easy deployment on Streamlit Cloud or local host
* Encodes categorical data automatically
* Includes scalable synthetic dataset for retraining

---

## 🖥️ How to Run Locally

### 1️⃣ Clone this repository

```bash
git clone https://github.com/yourusername/loan-approval-predictor.git
cd loan-approval-predictor
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501).

---

##  Sample Input

| Feature                  | Example Value |
| ------------------------ | ------------- |
| Number of Dependents     | 2             |
| Annual Income            | 8,00,000      |
| Loan Amount              | 15,00,000     |
| Loan Term                | 20            |
| CIBIL Score              | 720           |
| Residential Assets Value | 5,00,000      |
| Commercial Assets Value  | 2,00,000      |
| Luxury Assets Value      | 1,00,000      |
| Bank Asset Value         | 50,000        |
| Self Employed            | No            |
| Education                | Graduate      |

**Expected Output:**
 Loan Approved with ~94% confidence

---

##  Model Improvement Ideas

* Add feature engineering (Loan-to-Income ratio, Debt-to-Asset ratio)
* Tune hyperparameters using GridSearchCV
* Add more diverse data samples
* Integrate explainability (SHAP or LIME)

---

##  Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.

---

---

##  Acknowledgements

* Dataset inspired by publicly available **synthetic financial data**


---

> “42 is the answer to life, the universe, and reproducibility.” 
