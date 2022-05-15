import pytest
from app.data_preprocess import preprocess_data
from load_and_clean_data import process_data
from app.model_functions import split_data
from joblib import load
from app.model_functions import model_predictions


@pytest.fixture
def data():
    """
    Obtain data
    """
    df = process_data()
    return df


def test_null(data):
    """
    Check data has no null values
    """
    df = process_data()
    assert df.shape == df.dropna().shape


def test_balanced_data(data):
    """
    check if the target column is balanced
    """
    _, y_res, _ = preprocess_data(data, label='converted', training=True)
    assert y_res.value_counts()[0] == y_res.value_counts()[1]


def test_process_train(data):
    """
    Check train data has same number of rows for X and y
    """
    train, _ = split_data(data)
    X_train, y_train, _ = preprocess_data(
        train, label='converted', training=True)
    assert X_train.shape[0] == y_train.shape[0]


def test_process_test(data):
    """
    Check test data has same number of rows for X and y
    """
    ohe = load("functions/ohe.joblib")
    _, test = split_data(data)
    X_test, y_test, ohe = preprocess_data(
        test, label='converted', training=False, ohe=ohe)
    assert X_test.shape[0] == y_test.shape[0]


def test_predictions(data):
    """
  #  Check test data has same number of rows as predictions for evaluation
   # """
    model = load("functions/model.joblib")
    ohe = load("functions/ohe.joblib")
    _, test = split_data(data)
    X_test, y_test, _ = preprocess_data(
        test, label='converted', training=False, ohe=ohe)
    predictions = model_predictions(X_test, model)
    assert len(y_test) == len(predictions)
