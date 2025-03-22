import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import base64
import os

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
image_path = "background.jpg"  # Ensure this file exists in the same directory
if os.path.exists(image_path):
    set_background(get_base64_image(image_path))

# Load the trained model
model_path = "train_price_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
else:
    st.error("🚨 Model file 'train_price_model.pkl' not found! Please train and save the model first.")
    st.stop()

# Load label encoders
label_encoders = joblib.load("label_encoders.pkl") if os.path.exists("label_encoders.pkl") else {}

# Predefined lists
# Predefined lists (must match those used in training)
origins = ["madrid", "barcelona", "valencia", "seville", "bilbao"]
destinations = ["madrid", "barcelona", "valencia", "seville", "bilbao"]
train_types = ["ave", "alvia", "intercity", "regional", "av city", "md-ld", "ld", "ave-tge", "ave-md",
               "r. expres", "ave-ld", "ld-md", "trenhotel", "md-ave", "md", "ld-ave"]
train_classes = ["turista", "preferente", "club", "turista plus", "turista con enlace", "cama turista", "cama g. clase"]
fare_types = ["promo", "flexible", "adulto ida", "promo +", "individual flexible", "mesa", "grupos ida"]


# Streamlit App
st.title("🚆 Train Ticket Price Predictor")

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

    # Apply Label Encoding
    categorical_cols = ["origin", "destination", "train_type", "train_class", "fare"]
    for col in categorical_cols:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform([input_data[col].values[0]])[0]
        else:
            st.error(f"🚨 Error: '{col}' encoder is missing. Please retrain and save label encoders.")
            st.stop()

    # Make Prediction
    predicted_price = model.predict([input_data.iloc[0]])  # Convert to list

    # Show Result
    st.success(f"Estimated Ticket Price: €{predicted_price[0]:.2f}")
