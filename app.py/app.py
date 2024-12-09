import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('random_forest_model.pkl')

# Title and description
st.title("Payment Method Prediction")
st.write("Masukkan data pelanggan untuk memprediksi metode pembayaran:")

# Input fields
gender = st.selectbox("Gender:", ["Male", "Female"])
age = st.number_input("Age:", min_value=0, step=1)
product_category = st.selectbox(
    "Product Category:", ["Clothing", "Electronics", "Beauty"]
)
city = st.selectbox(
    "City:",
    ["Jakarta", "Bandung", "Surabaya", "Medan", "Yogyakarta", "Denpasar", "Semarang"],
)
quantity = st.number_input("Quantity:", min_value=1, step=1)
price_per_unit = st.number_input("Price per Unit:", min_value=0.0, step=0.01)

# Prediction button
if st.button("Predict"):
    # Preprocess input
    features = np.array([[ 
        age,
        quantity,
        price_per_unit,
        quantity * price_per_unit,
        1 if gender == "Male" else 0,
        1 if product_category == "Clothing" else 0,
        1 if product_category == "Electronics" else 0,
        1 if city == "Denpasar" else 0,
        1 if city == "Jakarta" else 0,
        1 if city == "Makassar" else 0,
        1 if city == "Medan" else 0,
        1 if city == "Semarang" else 0,
        1 if city == "Surabaya" else 0,
        1 if city == "Yogyakarta" else 0,
    ]])

    # Predict
    prediction = model.predict(features)[0]
    payment_methods = {0: "Cash", 1: "E-wallet", 2: "Credit Card"}
    result = payment_methods[prediction]

    # Display result
    st.success(f"Prediksi Metode Pembayaran: {result}")
