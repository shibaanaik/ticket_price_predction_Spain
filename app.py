import streamlit as st
import json
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import base64

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
set_background(get_base64_image(image_path))

# Load model parameters from JSON file
with open("train_price_model_params.json", "r") as f:
    params = json.load(f)

# Rebuild the Random Forest model using saved parameters
model = RandomForestRegressor(**params)

# Load label encoders from JSON
with open("label_encoders.json", "r") as f:
    label_classes = json.load(f)

# Rebuild the label encoders
label_encoders = {}
for col, classes in label_classes.items():
    le = LabelEncoder()
    le.classes_ = np.array(classes)  # Restore encoder classes
    label_encoders[col] = le

print("✅ Label encoders successfully loaded!")

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
        input_data[col] = input_data[col].astype(str).str.strip().str.lower()  # Ensure consistency
        encoder_classes = [str(cls).lower() for cls in label_encoders[col].classes_]

        
        if input_data[col].values[0] in encoder_classes:
            input_data[col] = label_encoders[col].transform([input_data[col].values[0]])[0]
        else:
            st.error(f"🚨 Error: '{input_data[col].values[0]}' is not in the trained categories for '{col}'. Please select a valid option.")
            st.stop()

    # Make Prediction
    predicted_price = model.predict(input_data)

    # Show Result
    st.success(f"Estimated Ticket Price: €{predicted_price[0]:.2f}")
