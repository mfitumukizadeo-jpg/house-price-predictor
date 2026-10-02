import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="house-price-predictor", page_icon="🏠")
st.title("house-price-predictor")
st.write("Enter the house characteristics to estimate the sale price in million RWF.")

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.sav")

model = load_model()

# Form inputs
area = st.number_input("Area (m²)", 26.0, 426.4, 105.7)
bedrooms = st.number_input("Bedrooms", 1, 6, 3, step=1)
bathrooms = st.number_input("Bathrooms", 1, 5, 3, step=1)
age = st.number_input("House Age (years)", 0.4, 49.5, 6.8)
distance = st.number_input("Distance to City Centre (km)", 0.07, 27.40, 3.27)
parking = st.number_input("Parking Spaces", 0, 3, 1, step=1)

if st.button("Predict"):
    # Pass ONLY the 6 numeric features the model was trained on
    row = pd.DataFrame([{
        "Area_m2": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "House_Age_Years": age,
        "Distance_to_City_km": distance,
        "Parking_Spaces": parking
    }])
    
    price = model.predict(row)[0]
    st.success(f"Predicted house price: {price:.2f} million RWF")