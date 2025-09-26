# ui/dashboard.py
import streamlit as st
import requests
import pandas as pd

API = st.text_input("API base URL", "http://127.0.0.1:8000")
asset = st.selectbox("Asset", ["BTC","GOLD","CRUDE","RELIANCE"])
if st.button("Get prediction"):
    r = requests.get(f"{API}/predict", params={"asset": asset})
    if r.status_code!=200:
        st.error(r.text)
    else:
        data = r.json()
        st.write("Prediction:", data["prediction"])
        st.metric("Probability Up", f"{data['prediction']['prob_up']:.2f}")
        st.subheader("Features")
        st.json(data["features"])
        st.subheader("News Headlines")
        for h in data["news"]["headlines"]:
            st.write("-", h)
