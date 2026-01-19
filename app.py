import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load model and encoders
model = pickle.load(open("CatBoost.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Background image
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1604467715878-83e0c3e2be01?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Page setup
st.set_page_config(page_title="Agri Price Predictor", layout="centered")
st.title("🌾 Telangana Agri Price Predictor")
st.markdown("Predict **Modal Price (₹ per quintal)** for Agricultural Commodities Across Market Yards")

# User inputs
yard = st.selectbox("Yard Name", label_encoders['YardName'].classes_)
comm = st.selectbox("Commodity Name", label_encoders['CommName'].classes_)
variety = st.selectbox("Variety Type", label_encoders['VarityName'].classes_)
date = st.date_input("Select Date", datetime.today())

# Encode inputs
yard_enc = label_encoders['YardName'].transform([yard])[0]
comm_enc = label_encoders['CommName'].transform([comm])[0]
variety_enc = label_encoders['VarityName'].transform([variety])[0]

# Extract date features
year, month, day = date.year, date.month, date.day

# Dummy values for Min/Max (to be replaced with better estimates if needed)
min_price = 1000
max_price = 2000
price_range = max_price - min_price
avg_price = (min_price + max_price) / 2

# Feature vector
features = np.array([[yard_enc, comm_enc, variety_enc,
                      min_price, max_price, price_range, avg_price,
                      year, month, day]])

# Predict
if st.button("Predict Modal Price"):
    price = model.predict(features)[0]
    st.success(f"🌟 Predicted Modal Price: ₹{price:.2f} per quintal")
