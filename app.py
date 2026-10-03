import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rwanda House Price Predictor")

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.joblib")  # the Pipeline (preprocessing + regression)

model = load_model()

NEIGHBORHOODS = ["Gasabo", "Huye", "Kicukiro", "Kigali City", "Musanze", "Nyarugenge"]

st.title("Rwanda House Price Predictor")
st.write("Enter the house characteristics to estimate the sale price in million RWF.")

area = st.number_input("Area (m²)", min_value=20.0, max_value=500.0, value=120.0)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
age = st.number_input("House Age (years)", min_value=0.0, max_value=60.0, value=5.0)
distance = st.number_input("Distance to City Centre (km)", min_value=0.0, max_value=30.0, value=5.0)
parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1, step=1)
neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS)

if st.button("Predict price"):
    row = pd.DataFrame([{
        "Area_m2": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "House_Age_Years": age,
        "Distance_to_City_km": distance,
        "Parking_Spaces": parking,
        "Neighborhood": neighborhood,
    }])
    price = model.predict(row)[0]
    st.success(f"Estimated price: {price:,.1f} million RWF")
    st.caption("Estimate from a small dataset (113 houses); typical error is about 18 million RWF.")
