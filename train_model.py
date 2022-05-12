import logging
from functions.data_preprocess import preprocess_data
from functions.model_functions import split_data,train_model,model_predictions,compute_metrics
from load_and_clean_data import process_data
from joblib import dump

logging.basicConfig(
    filename='./outputs/process.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    df = process_data()
    train, test=split_data(df)
    X_train,y_train = preprocess_data(train)
    X_test,y_test = preprocess_data(test)
    model = train_model(X_train, y_train)
    dump(model, './outputs/model.joblib')
    predictions = model_predictions(X_test, model)
    accuracy,precision, recall, f1 = compute_metrics(y_test, predictions)
    


