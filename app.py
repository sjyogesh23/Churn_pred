import streamlit as st
import pandas as pd
import pickle
import webbrowser

with open('churn_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
loaded_encoder = model_data["encoder"]

st.title("Telco Customer Churn Prediction")

st.subheader("Personal Information")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])

with col2:
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

st.subheader("Tenure & Service Details")
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, step=1)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

st.subheader("Internet & Support Services")
col1, col2 = st.columns(2)

with col1:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

st.subheader("Contract & Payment Information")
col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    total_charges = st.number_input("Total Charges", min_value=0.0)

with col2:
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

st.write(' ')
submitted = st.button("Predict", use_container_width=True)

if submitted:
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    ip_df = pd.DataFrame([input_data])

    for col, encoder in loaded_encoder.items():
        ip_df[col] = encoder.transform(ip_df[col])

    prediction = loaded_model.predict(ip_df)[0]
    probability = loaded_model.predict_proba(ip_df)[0]

    if prediction == 1:
        st.error("Prediction: Yes, the customer is likely to churn.")
    else:
        st.success("Prediction: No, the customer is not likely to churn.")

    st.info(f"Prediction Probability: {probability}")

st.write(' ')

left, right = st.columns(2)

if left.button("Code Link", use_container_width=True):
    webbrowser.open("https://github.com/sjyogesh23/Churn_pred")

if right.button("Developer Portfolio", use_container_width=True):
    webbrowser.open("https://yogeshsj.vercel.app/")