import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def preprocess_data(df):
   
    #creating X and y           
    X = df.drop(columns='converted')
    y = df['converted']

    #one hot encoding the X column
    X = X=pd.get_dummies(X) 

    #balancing the data 
    smote=SMOTE(random_state=0)
    X_res, y_res = smote.fit_resample(X, y)

    return X_res, y_res

