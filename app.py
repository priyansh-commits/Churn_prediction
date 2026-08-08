
import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Load trained model
# -----------------------------------

model_data = joblib.load("churn_model.pkl")

model = model_data["model"]
features = model_data["features"]


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Telecom Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn.")


# -----------------------------------
# Customer Information
# -----------------------------------

st.header("Customer Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=600.0
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

tech_support = st.selectbox(
    "Tech Support",
    [0, 1]
)

contract_type = st.selectbox(
    "Contract Type",
    ["Month-to-Month", "One-Year", "Two-Year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber Optic", "No"]
)


# -----------------------------------
# Prediction
# -----------------------------------

if st.button("🔮 Predict Churn"):

    # Create input dataframe using original values
    input_data = pd.DataFrame({
        "Age": [age],
        "Tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "TechSupport": [tech_support],
        "Gender": [gender],
        "ContractType": [contract_type],
        "InternetService": [internet_service]
    })

    # -----------------------------------
    # Create dummy variables
    # -----------------------------------

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "Gender",
            "ContractType",
            "InternetService"
        ],
        drop_first=True
    )

    # -----------------------------------
    # Match training features
    # -----------------------------------

    input_data = input_data.reindex(
        columns=features,
        fill_value=0
    )

    # -----------------------------------
    # Prediction
    # -----------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # -----------------------------------
    # Display result
    # -----------------------------------

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is unlikely to churn")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )
```
