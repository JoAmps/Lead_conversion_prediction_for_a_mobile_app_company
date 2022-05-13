import pytest
from functions.data_preprocess import preprocess_data
from load_and_clean_data import process_data
from functions.model_functions import split_data, model_predictions
from joblib import load


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
    _, y_res = preprocess_data(data)
    assert y_res.value_counts()[0] == y_res.value_counts()[1]


def test_process_train(data):
    """
    Check train data has same number of rows for X and y
    """
    train, _ = split_data(data)
    X_train, y_train = preprocess_data(train)
    assert X_train.shape[0] == y_train.shape[0]


def test_process_test(data):
    """
    Check test data has same number of rows for X and y
    """
    _, test = split_data(data)
    X_test, y_test = preprocess_data(test)
    assert X_test.shape[0] == y_test.shape[0]


def test_predictions(data):
    """
    Check test data has same number of rows as predictions for evaluation
    """
    model = load("outputs/model.joblib")
    _, test = split_data(data)
    X_test, y_test = preprocess_data(test)
    predictions = model_predictions(X_test, model)
    assert len(y_test) == len(predictions)
