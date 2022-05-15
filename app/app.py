from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from pandas.core.frame import DataFrame
import numpy as np
from data_preprocess import preprocess_data
from model_functions import inference



class User(BaseModel):
    lead_utm_source: Literal[
        'facebook', 'twitter', 'google_ads', 'youtube', 'newsletter','medium','instagram']
    lead_utm_medium: Literal[
        'social', 'paid', 'email','affiliates','banner','cpc','organic_search','display'
    ]
    lead_weekday_of_registration: Literal[
        'weekday', 'weekend'
    ]
    #lead_month_day_of_registration: Literal[
    #    '1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0','11.0','12.0',
     #   '13.0','14.0','15.0','16.0','17.0','18.0','19.0','20.0','21.0','22.0','23.0','24.0','25.0','26.0',
     #   '27.0','28.0','29.0','30.0','31.0'
    #]
    lead_country_of_registration: Literal[
        'usa', 'uk' ,'india', 'france','russia','china','south_africa','germany','nigeria','japan','italy','pakistan','ghana'
    ]
    lead_ua_device_class: Literal[
        'Phone', 'Desktop', 'Tablet','Set-top box','Mobile','TV'
    ]

    lead_time_of_registration: Literal[
        'Dawn', 'Morning', 'Afternoon','Evening','Night'
    ]

    Time_since_registration: Literal['Under one day',
                          'A month to over a year',
                          'A week to a month',
                          'A day to a week']


app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "Welcome to the Lead conversion webpage"}


@app.post("/")
async def inferences(user_data: User):
    model_object = load("../functions/model.joblib")
    ohe = load("../functions/ohe.joblib")
    array = np.array([[
                     user_data.lead_utm_source,
                     user_data.lead_utm_medium,
                     user_data.lead_weekday_of_registration,
                     user_data.lead_country_of_registration,
                     user_data.lead_ua_device_class,
                     user_data.lead_time_of_registration,
                     user_data.Time_since_registration,
                     ]])

    df_temp = DataFrame(data=array, columns=[
        'lead_utm_source',
        'lead_utm_medium',
        'lead_weekday_of_registration',
        'lead_month_day_of_registration',
        'lead_ua_device_class',
        'lead_time_of_registration',
        'Time_since_registration',
    ])
    X,_,_ = preprocess_data(df_temp,ohe=ohe)
    prediction = inference(model_object, X)
    prediction_label = ['This lead wont convert' if label == 0 else 'This lead would convert' for label in prediction ]
    # Return response back to client
    return {"prediction": prediction_label}
    #y = (prediction.tolist())
    #return {"prediction": y}#