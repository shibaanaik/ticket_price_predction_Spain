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
image_path = "background.jpg"
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
# Define categorical columns before using them
categorical_cols = ["origin", "destination", "train_type", "train_class", "fare"]

for col in categorical_cols:
    if col in label_encoders:
        try:
            input_data[col] = label_encoders[col].transform([input_data[col].values[0]])[0]
        except ValueError:
            st.warning(f"⚠️ Warning: The value '{input_data[col].values[0]}' for '{col}' was not seen during training. Assigning a default value.")
            input_data[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]  # Assign first known value
    else:
        st.error(f"🚨 Error: '{col}' encoder is missing in 'label_encoders.pkl'. Retrain the model with all categorical columns encoded.")
        st.stop()

# Predefined lists (must match those used in training)
origins = ["Madrid", "Barcelona", "Valencia", "Seville", "Bilbao"]
destinations = ["Madrid", "Barcelona", "Valencia", "Seville", "Bilbao"]
train_types = ["AVE", "Alvia", "Intercity", "Regional", "AV City", "MD-LD", "LD", "AVE-TGV", "AVE-MD",
               "R. EXPRES", "AVE-LD", "LD-MD", "TRENHOTEL", "MD-AVE", "MD", "LD-AVE"]
train_classes = ["Turista", "Preferente", "Club", "Turista Plus", "Turista con enlace", "Cama Turista", "Cama G. Clase"]
fare_types = ["Promo", "Flexible", "Adulto ida", "Promo +", "Individual Flexible", "Mesa", "Grupos Ida"]
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
            st.error(f"Error: '{col}' encoder is missing in 'label_encoders.pkl'. Retrain the model with all categorical columns encoded.")
            st.stop()

    # Convert input to numpy array and reshape
    input_array = input_data.values.reshape(1, -1)

    # Make Prediction
    predicted_price = model.predict(input_array)

    # Show Result
    st.success(f" Estimated Ticket Price: €{predicted_price[0]:.2f}")
