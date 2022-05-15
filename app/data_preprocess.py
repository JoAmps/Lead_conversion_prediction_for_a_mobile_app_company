import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer


def preprocess_data(df, label=None,training=False, ohe=None):
    if label is not None:
        y = df[label]
        X = df.drop([label], axis=1)
    else:
        X = df
        y = np.array([])
    
    if training is True:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X = ohe.fit_transform(X)
    else:
        X = ohe.transform(X)

    return X, y,ohe
