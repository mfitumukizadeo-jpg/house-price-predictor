
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# App title and description
st.title("🏠 House Price Predictor")
st.write(
    "Enter the house characteristics to estimate "
    "the sale price in million RWF."
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.sav")

model = load_model()

# User inputs
area = st.number_input(
    "Area (m²)", min_value=26.0, max_value=426.4, value=105.7
)

bedrooms = st.number_input(
    "Bedrooms", min_value=1, max_value=6, value=3, step=1
)

bathrooms = st.number_input(
    "Bathrooms", min_value=1, max_value=5, value=3, step=1
)

age = st.number_input(
    "House Age (years)", min_value=0.4, max_value=49.5, value=6.8
)

distance = st.number_input(
    "Distance to City Centre (km)",
    min_value=0.07, max_value=27.40, value=3.27
)

parking = st.number_input(
    "Parking Spaces", min_value=0, max_value=3, value=1, step=1
)

neighborhood = st.selectbox(
    "Neighborhood",
    [
        "Gasabo",
        "Huye",
        "Kicukiro",
        "Kigali City",
        "Musanze",
        "Nyarugenge"
    ]
)

# Prediction
if st.button("Predict"):
    # Create input DataFrame
    row = pd.DataFrame([{
        "Area_m2": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "House_Age_Years": age,
        "Distance_to_City_km": distance,
        "Parking_Spaces": parking,
        "Neighborhood": neighborhood
    }])

    try:
        # Match the model's expected column order
        if hasattr(model, "feature_names_in_"):
            row = row[model.feature_names_in_]

        # Predict house price
        price = model.predict(row)[0]

        # Display result
        st.success(
            f"Predicted house price: {price:.2f} million RWF"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
```
