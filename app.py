import streamlit as st
import pandas as pd
import joblib
import json
import os

# ---------------------------
# 🌍 Earthquake Alert Prediction App
# ---------------------------
st.set_page_config(page_title="Earthquake Alert Prediction", layout="centered")

st.title("🌍 Earthquake Alert Prediction App")
st.markdown(
    """
    This app uses a trained **Machine Learning model** to predict earthquake alert levels.  
    Enter the earthquake parameters below to get the prediction.
    """
)

# ---------------------------
# Load Model and Preprocessors
# ---------------------------
model_path = "model/rf_model.joblib"
imputer_path = "model/imputer.joblib"
scaler_path = "model/scaler.joblib"
meta_path = "model/metadata.json"

# Check if model files exist
if not all(os.path.exists(p) for p in [model_path, imputer_path, scaler_path, meta_path]):
    st.error("⚠️ Model files not found! Please run `train_model.py` first to generate the model.")
    st.stop()

# Load model and metadata
model = joblib.load(model_path)
imputer = joblib.load(imputer_path)
scaler = joblib.load(scaler_path)

with open(meta_path, "r") as f:
    meta = json.load(f)

features = meta["features"]
target = meta["target"]

# ---------------------------
# User Input Section
# ---------------------------
st.subheader("🧾 Input Earthquake Parameters")

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔮 Predict"):
    input_df = pd.DataFrame([user_input])
    
    # Apply preprocessing in same order as training
    input_df = pd.DataFrame(imputer.transform(input_df), columns=features)
    input_df = pd.DataFrame(scaler.transform(input_df), columns=features)
    
    prediction = model.predict(input_df)[0]
    
    st.success(f"🚨 Predicted Alert Level: **{prediction}**")

    st.markdown("---")
    st.caption("Model trained using Random Forest Classifier")

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
    ---
    **Developed by:** Your Name  
    **Dataset:** Earthquake Alert Prediction Dataset (Kaggle)
    """
)
