import logging
from functions.data_preprocess import preprocess_data
from functions.model_functions import split_data, train_model,\
     model_predictions, compute_metrics, plot_visualizations
from load_and_clean_data import process_data
from joblib import dump


logging.basicConfig(
    filename='./outputs/process.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    df = process_data()
    train, test = split_data(df)
    X_train, y_train = preprocess_data(train)
    X_test, y_test = preprocess_data(test)
    model = train_model(X_train, y_train)
    dump(model, './outputs/model.joblib')
    predictions = model_predictions(X_test, model)
    accuracy, precision, recall, f1 = compute_metrics(y_test, predictions)
    model_scores = []
    scores = "accuracy: %s " "precision: %s " \
        "recall: %s f1: %s" % (accuracy, precision, recall, f1)
    model_scores.append(scores)
    with open('./outputs/model_metrics.txt', 'w') as out:
        for score in model_scores:
            out.write(score)

    plot_visualizations(predictions, y_test)
