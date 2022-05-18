import streamlit as st
import requests


def run():
    st.write("""

    # Lead Conversion Prediction Webpage

    Lead conversion refers to converting an individual that shows
    interest in your business(in this case app) into a paying customer.
    Its important to determine such individuals, and very important
    to predict which of your leads are likely to convert.
    Companies are interested in targeting these leads that are likely to
    convert. These leads can  be targeted  with special deals and
    promotions to influence  them to stay with the company.
    This app predicts if a lead would convert. Lead converting means the lead
    succesfully registered with the app

    """)
    lead_utm_source = st.selectbox(
        "Select lead utm source",
        ('facebook',
         'twitter',
         'google_ads',
         'youtube',
         'newsletter',
         'medium',
         'instagram'))
    lead_utm_medium = st.selectbox(
        "Select lead utm medium",
        ('social',
         'paid',
         'email',
         'affiliates',
         'banner',
         'cpc',
         'organic_search',
         'display'))
    lead_weekday_of_registration = st.selectbox(
        "Select if lead registered on a weekday/weekend",
        ('weekday', 'weekend'))
    lead_country_of_registration = st.selectbox(
        "Select Country of origin of lead",
        ('usa',
         'uk',
         'india',
         'france',
         'russia',
         'china',
         'south_africa',
         'germany',
         'nigeria',
         'japan',
         'italy',
         'pakistan',
         'ghana'))
    lead_ua_device_class = st.selectbox(
        "Select the device is using",
        ('Phone',
         'Desktop',
         'Tablet',
         'Set-top box',
         'Mobile',
         'TV'))
    lead_time_of_registration = st.selectbox(
        "Select Time of day lead registered",
        ('Dawn',
         'Morning',
         'Afternoon',
         'Evening',
         'Night'))
    Time_since_registration = st.selectbox(
        "Select time passed since lead checked out app",
        ('Under one day',
         'A month to over a year',
         'A week to a month',
         'A day to a week'))

    data = {
        'lead_utm_source': lead_utm_source,
        'lead_utm_medium': lead_utm_medium,
        'lead_weekday_of_registration': lead_weekday_of_registration,
        'lead_country_of_registration': lead_country_of_registration,
        'lead_ua_device_class': lead_ua_device_class,
        'lead_time_of_registration': lead_time_of_registration,
        'Time_since_registration': Time_since_registration
    }

    if st.button("Predict"):
        response = requests.post("http://127.0.0.1:8080/predict", json=data,timeout=5)
        prediction = response.text
        st.success(f"The prediction from model: {prediction}")


if __name__ == '__main__':
    run()
