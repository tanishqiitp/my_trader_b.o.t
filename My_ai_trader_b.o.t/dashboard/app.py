import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("Market Prediction Dashboard")

asset = st.text_input("Enter Asset (e.g. BTC, GOLD, CRUDE, TCS):", "TCS")
if st.button("Get Prediction"):
    response = requests.get(f"{API_URL}/predict", params={"asset": asset})
    if response.status_code == 200:
        data = response.json()
        st.json(data)
    else:
        st.error("API error")
