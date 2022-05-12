import pytest
from functions.data_preprocess import process_data
import pandas as pd
from load_data import load_data
#def load_preprocess():
#    df = pd.read_csv('datasets/lead_convert.csv',index_col=0)
 #   _,_,df = process_data(df)
 #   return df

@pytest.fixture
def data():
    """
    Obtain data
    """
    df = load_data()
    return df

def test_null(data):
    """
    Check data has no null values
    """
    _,_,df=process_data(data)
    assert df.shape == df.dropna().shape

def test_balanced_data(data):
    """
    check if the target column is balanced
    """
    _,y_res,_ = process_data(data)
    assert y_res.value_counts()[0]==y_res.value_counts()[1]     

def test_process_train(data):
    """
    Check train data has same number of rows for X and y
    """
    X_res,y_res,_ = process_data(data)
    assert X_res.shape[0] == y_res.shape[0]           