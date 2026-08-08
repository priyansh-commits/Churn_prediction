import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model
# -----------------------------

model_data = joblib.load("churn_model.pkl")

model = model_data["model"]
features = model_data["features"]

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Telecom Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn.")

# -----------------------------
# Inputs
# -----------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    value=12
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# Add your categorical inputs here
# Example:
#
# contract = st.selectbox(
#     "Contract",
#     ["Month-to-month", "One year", "Two year"]
# )

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Churn"):

    # Create dataframe from user input
    input_data = pd.DataFrame({
        "Age": [age],
        "Tenure": [tenure],
        "TotalCharges": [total_charges],

        # Add your categorical columns here
    })

    # One-hot encode
    input_data = pd.get_dummies(input_data)

    # Make input match training features exactly
    input_data = input_data.reindex(
        columns=features,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is unlikely to churn")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )
