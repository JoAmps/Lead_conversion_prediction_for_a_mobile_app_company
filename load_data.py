import pandas as pd

def load_data():
    df = pd.read_csv('datasets/lead_convert.csv',index_col=0)
    return df