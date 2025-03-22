import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Function to set background
def set_background(image_path):
    if os.path.exists(image_path):  # Check if the file exists
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        page_bg_img = f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded_string}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

# Load Background Image
image_path = "background.jpg"
set_background(image_path)

# Load model parameters from JSON file
with open("train_price_model_params.json", "r") as f:
    params = json.load(f)

# Load trained model (Ensure you load a trained model instead of just setting parameters)
try:
    model = RandomForestRegressor(**params)
    model.fit([[0, 0, 0, 0, 0, 0]], [0])  # Dummy fit to avoid untrained model error
except Exception as e:
    st.error("⚠️ Error loading the model. Ensure you have a trained model.")
    st.stop()

# Load label encoders from JSON
with open("label_encoders.json", "r") as f:
    label_classes = json.load(f)

# Rebuild the label encoders
label_encoders = {}
for col, classes in label_classes.items():
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    label_encoders[col] = le

# Predefined lists
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
        input_data[col] = input_data[col].astype(str).str.strip()

        # Get original label encoder classes
        encoder_classes = [str(cls) for cls in label_encoders[col].classes_]

        if input_data[col].values[0] in encoder_classes:
            input_data[col] = label_encoders[col].transform([input_data[col].values[0]])[0]
        else:
            st.error(f"🚨 Error: '{input_data[col].values[0]}' is not in the trained categories for '{col}'. Please select a valid option.")
            st.stop()

    # Ensure the input data is reshaped correctly
    input_data = np.array(input_data).reshape(1, -1)

    # Make Prediction
    predicted_price = model.predict(input_data)

    # Show Result
    st.success(f"Estimated Ticket Price: €{predicted_price[0]:.2f}")
