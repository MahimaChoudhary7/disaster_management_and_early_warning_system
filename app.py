import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ========== Load the Trained Model ==========
model_path = "model/model.pkl"

if not os.path.exists(model_path):
    st.error("⚠️ Model not found! Please run train_model.py first.")
else:
    model = joblib.load(model_path)
    st.success("✅ Disaster Prediction Model Loaded Successfully!")

# ========== App Title ==========
st.title("🌪️ Machine Learning-Based Disaster Prediction System")
st.write("Enter environmental data below to predict if a disaster alert should be raised.")

# ========== Input Fields ==========
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=60.0, value=30.0)
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
    region = st.selectbox("📍 Region", ["North", "South", "East", "West"])

with col2:
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    wind_speed = st.number_input("🌬️ Wind Speed (km/h)", min_value=0.0, max_value=200.0, value=40.0)

# ========== Prepare Input Data ==========
# ========== Prepare Input Data (Fixed) ==========

# Create dummy variables for all known regions — make sure all four are present
input_data = {
    "temperature": [temperature],
    "humidity": [humidity],
    "rainfall": [rainfall],
    "wind_speed": [wind_speed],
    "region_North": [1 if region == "North" else 0],
    "region_South": [1 if region == "South" else 0],
    "region_East":  [1 if region == "East"  else 0],
    "region_West":  [1 if region == "West"  else 0]
}

input_df = pd.DataFrame(input_data)

# Match columns with those seen during model training
expected_cols = model.feature_names_in_  # only works if model was trained with feature names
for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0  # add missing ones as 0
input_df = input_df[expected_cols]


# ========== Predict Button ==========
if st.button("🔍 Predict Disaster Alert"):
    if not os.path.exists(model_path):
        st.warning("Please train the model first by running train_model.py")
    else:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        if prediction == 1:
            st.error(f"🚨 Disaster Alert! Probability: {probability:.2f}%")
        else:
            st.success(f"✅ No Disaster Expected. Safety Level: {100 - probability:.2f}%")

# ========== Footer ==========
st.markdown("---")
st.caption("Developed by Mahima Choudhary | Advanced Python Programming Mini Project")
