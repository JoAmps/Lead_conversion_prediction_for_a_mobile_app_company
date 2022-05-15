from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from pandas.core.frame import DataFrame
import numpy as np
import uvicorn
from data_preprocess import preprocess_data
from model_functions import inference


class User(BaseModel):
    lead_utm_source: Literal['facebook',
                             'twitter',
                             'google_ads',
                             'youtube',
                             'newsletter',
                             'medium',
                             'instagram']
    lead_utm_medium: Literal['social',
                             'paid',
                             'email',
                             'affiliates',
                             'banner',
                             'cpc',
                             'organic_search',
                             'display']
    lead_weekday_of_registration: Literal[
        'weekday', 'weekend'
    ]

    lead_country_of_registration: Literal['usa',
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
                                          'ghana']
    lead_ua_device_class: Literal[
        'Phone', 'Desktop', 'Tablet', 'Set-top box', 'Mobile', 'TV'
    ]

    lead_time_of_registration: Literal[
        'Dawn', 'Morning', 'Afternoon', 'Evening', 'Night'
    ]

    Time_since_registration: Literal['Under one day',
                                     'A month to over a year',
                                     'A week to a month',
                                     'A day to a week']


app = FastAPI()


@app.get("/")
def home():
    """
     Testing availiabilty of the application.
     """
    return {'message': 'App works!'}


@app.post("/predict")
def inferences(user_data: User):
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
        'lead_country_of_registration',
        'lead_ua_device_class',
        'lead_time_of_registration',
        'Time_since_registration',
    ])
    X, _, _ = preprocess_data(df_temp, ohe=ohe)
    prediction = inference(model_object, X)
    prediction_label = ['This lead wont convert' if label ==
                        0 else 'Lead will convert' for label in prediction]
    return {"prediction": prediction_label}


if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
