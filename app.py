import streamlit as st
import json
import base64
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Function to set background
def set_background(image_file):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_file}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Load and encode background image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load Background Image
image_path = "background.jpg"  # Ensure this file exists
set_background(get_base64_image(image_path))

# Load model parameters from JSON file
with open("train_price_model_params.json", "r") as f:
    params = json.load(f)

# Load label mappings from JSON
with open("label_mappings.json", "r") as f:
    label_mappings = json.load(f)

# Rebuild the Random Forest model using saved parameters
model = RandomForestRegressor(**params)

st.title("🚆 Train Ticket Price Predictor")

# Predefined lists (must match those used in training)
origins = ["madrid", "barcelona", "valencia", "seville", "silbao"]
destinations = ["madrid", "barcelona", "valencia", "seville", "bilbao"]
train_types = ["ave", "alvia", "intercity", "regional", "av city", "md-ld", "ld", "ave-tge", "ave-md",
               "r. expres", "ave-ld", "ld-md", "trenhotel", "md-ave", "md", "ld-ave"]
train_classes = ["turista", "preferente", "club", "turista plus", "turista con enlace", "cama turista", "cama g. clase"]
fare_types = ["promo", "flexible", "adulto ida", "promo +", "individual flexible", "mesa", "grupos ida"]

# User Inputs
origin = st.selectbox("Select Origin Station:", origins)
destination = st.selectbox("Select Destination Station:", destinations)
train_type = st.selectbox("Select Train Type:", train_types)
train_class = st.selectbox("Select Train Class:", train_classes)
fare = st.selectbox("Select Fare Type:", fare_types)
travel_duration = st.number_input("Enter Travel Duration (minutes):", min_value=10, max_value=1000)

# Predict Button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[origin, destination, train_type, train_class, fare, travel_duration]],
                              columns=["origin", "destination", "train_type", "train_class", "fare", "travel_duration"])

    # Apply Label Encoding using Dictionary Lookup
    categorical_cols = ["origin", "destination", "train_type", "train_class", "fare"]

    for col in categorical_cols:
        input_data[col] = input_data[col].astype(str).str.strip()  # Ensure consistent formatting
        
        if input_data[col].values[0] in label_mappings[col]:  
            input_data[col] = label_mappings[col][input_data[col].values[0]]  # Convert using dictionary
        else:
            st.error(f"🚨 Error: '{input_data[col].values[0]}' is not recognized in '{col}'. Available options: {list(label_mappings[col].keys())}")
            st.stop()

    # Make Prediction
    predicted_price = model.predict(input_data)

    # Show Result
    st.success(f"Estimated Ticket Price: €{predicted_price[0]:.2f}")
