import streamlit as st
from functions.data_preprocess import preprocess_data
from joblib import load

model = load("outputs/model.joblib")

@st.cache()

def prediction(lead_utm_source, lead_utm_medium):
    if lead_utm_source == "facebook":
        lead_utm_source = 0
    else:
        Gender = 1
