import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/crop_rf_model.joblib')

st.title("🌱 Smart Crop Recommendation System")
st.write("Enter the soil and environmental metrics below to get the best crop recommendation.")

col1, col2 = st.columns(2)
with col1:
    N = st.number_input("Nitrogen (N) Content in Soil", min_value=0, max_value=150, value=50)
    P = st.number_input("Phosphorus (P) Content in Soil", min_value=0, max_value=150, value=50)
    K = st.number_input("Potassium (K) Content in Soil", min_value=0, max_value=250, value=50)
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

if st.button("Predict Optimal Crop"):
    input_data = pd.DataFrame([[N, P, K, temp, humidity, ph, rainfall]], 
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    prediction = model.predict(input_data)[0]
    st.write("---")
    st.subheader("📊 What drove this prediction?")
    st.write("This chart shows which soil and environmental metrics the model prioritized overall when making its decision.")
    
    importances = model.feature_importances_
    
    feature_names = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        
    importance_df = pd.DataFrame({'Importance': importances}, index=feature_names)    
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
    st.bar_chart(importance_df)
    
    st.success(f"🎯 The most suitable crop for these conditions is: **{prediction.capitalize()}**")