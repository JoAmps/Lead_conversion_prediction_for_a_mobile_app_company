import logging
from functions.data_preprocess import preprocess_data
from functions.model_functions import split_data,train_model,model_predictions,compute_metrics,get_features,plot_visualizations
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
    model_scores = []
    scores = "accuracy: %s " "precision: %s " \
        "recall: %s f1: %s" % (accuracy,precision, recall, f1)
    model_scores.append(scores)
    with open('./outputs/model_metrics.txt', 'w') as out:
        for score in model_scores:
            out.write(score)

    plot_visualizations(predictions, y_test)         

    slice_values = []
    for feature in get_features():
        for cls in test[feature].unique():
            df_temp = test[test[feature] == cls]
            X_test_temp,y_test_temp=preprocess_data(test)
            y_preds = model.predict(X_test_temp)
            accuracy_temp,precision_temp, recall_temp, f1_temp = compute_metrics(y_test_temp, y_preds)
            results = "[%s->%s] Accuracy: %s  Precision: %s " \
                    "Recall: %s FBeta: %s" % (
                        feature,
                        cls,
                        accuracy_temp,
                        precision_temp,
                        recall_temp,
                        f1_temp)
            slice_values.append(results)

    with open('./outputs/slice_model_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')     

    

