import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np


def preprocess_data(df, label=None):
   
    if label is not None:
        y = df[label]
        X = df.drop([label], axis=1)
        smote = SMOTE(random_state=0)
        X = pd.get_dummies(X)
        X_res, y_res = smote.fit_resample(X, y)
    else:
        y_res = np.array([])
        X_res = pd.get_dummies(df)
    # creating X and y
    #X = df.drop(columns='converted')
    #y = df['converted']
    
    # one hot encoding the X column
    # balancing the data

    return X_res, y_res
