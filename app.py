import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("AQI_model (3).pkl")
scaler = joblib.load("scaler (3).pkl")  # Save this from your training notebook

st.title("🌫 Air Quality Index (AQI) Prediction App")
st.write("Enter the pollutant concentrations to predict AQI.")

# Inputs
pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, format="%.2f")
pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, format="%.2f")
no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, format="%.2f")
so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, format="%.2f")
co = st.number_input("CO (mg/m³)", min_value=0.0, format="%.2f")
ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, format="%.2f")

if st.button("Predict AQI"):
    # Arrange input in training order
    features = np.array([[pm25, pm10, no2, so2, co, ozone]])
    
    # Apply same scaling as during training
    features_scaled = scaler.transform(features)
    
    # Predict
    predicted_aqi = model.predict(features_scaled)[0]
    
    st.success(f"Predicted AQI: {predicted_aqi:.2f}")
