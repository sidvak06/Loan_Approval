import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("loan_model.pkl")

# Page title
st.title("Loan Approval Prediction")
st.write("Enter the applicant's information to predict the probability of loan approval.")

# -----------------------------
# User Inputs
# -----------------------------

no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=2
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    value=500000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=1000000
)

loan_term = st.number_input(
    "Loan Term (Years)",
    min_value=1,
    max_value=30,
    value=10
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=700
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0,
    value=1000000
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0,
    value=0
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0,
    value=500000
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0,
    value=500000
)

# -----------------------------
# Create Input DataFrame
# -----------------------------

input_data = pd.DataFrame({
    "no_of_dependents": [no_of_dependents],
    "education": [education],
    "self_employed": [self_employed],
    "income_annum": [income_annum],
    "loan_amount": [loan_amount],
    "loan_term": [loan_term],
    "cibil_score": [cibil_score],
    "residential_assets_value": [residential_assets_value],
    "commercial_assets_value": [commercial_assets_value],
    "luxury_assets_value": [luxury_assets_value],
    "bank_asset_value": [bank_asset_value]
})

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Loan Approval"):

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    # Your dataset contains leading whitespace
    approved_index = list(model.classes_).index(" Approved")

    approval_probability = probabilities[0][approved_index]
    approval_percentage = approval_probability * 100

    # Display probability
    st.metric(
        "Probability of Approval",
        f"{approval_percentage:.2f}%"
    )

    # Display prediction
    if prediction[0] == " Approved":
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

