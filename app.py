
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

model = joblib.load('model/churn_model.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to leave based on their profile.")

st.sidebar.header("Customer profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly charges ($)", 18.0, 120.0, 65.0)
total_charges = monthly_charges * tenure
avg_monthly_spend = total_charges / (tenure + 1)

contract = st.sidebar.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
paperless = st.sidebar.selectbox("Paperless billing", ["Yes", "No"])
online_security = st.sidebar.selectbox("Online security", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech support", ["Yes", "No"])
senior = st.sidebar.selectbox("Senior citizen", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has dependents", ["Yes", "No"])
partner = st.sidebar.selectbox("Has partner", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Phone service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple lines", ["Yes", "No"])
online_backup = st.sidebar.selectbox("Online backup", ["Yes", "No"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.sidebar.selectbox("Streaming movies", ["Yes", "No"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "AvgMonthlySpend": avg_monthly_spend,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner_Yes": 1 if partner == "Yes" else 0,
    "Dependents_Yes": 1 if dependents == "Yes" else 0,
    "PhoneService_Yes": 1 if phone_service == "Yes" else 0,
    "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
    "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
    "InternetService_No": 1 if internet == "No" else 0,
    "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
    "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,
    "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
    "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
    "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
    "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0,
    "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
    "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
    "gender_Male": 1 if gender == "Male" else 0,
}

# Add any missing columns the model expects
import numpy as np
model_features = model.get_booster().feature_names
for col in model_features:
    if col not in input_dict:
        input_dict[col] = 0

input_df = pd.DataFrame([input_dict])[model_features]

prob = model.predict_proba(input_df)[0][1]
prediction = model.predict(input_df)[0]

st.subheader("Prediction result")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Churn probability", f"{prob*100:.1f}%")
with col2:
    st.metric("Tenure", f"{tenure} months")
with col3:
    st.metric("Monthly charges", f"${monthly_charges:.2f}")

if prob > 0.6:
    st.error("High churn risk — this customer is likely to leave")
elif prob > 0.35:
    st.warning("Medium churn risk — worth monitoring")
else:
    st.success("Low churn risk — this customer is likely to stay")

st.subheader("What is driving this prediction?")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)
fig, ax = plt.subplots(figsize=(8, 4))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=model_features
    ),
    show=False
)
st.pyplot(fig)
plt.close()

st.subheader("Overall feature importance")
st.image("outputs/shap_importance.png")
