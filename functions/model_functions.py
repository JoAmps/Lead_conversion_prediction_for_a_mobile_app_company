from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import logging

def model_kfold(X_train, y_train):
    """
    Trains a machine learning model and returns the
    trained model.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    try:
        model = DecisionTreeClassifier(random_state=0)
        k = 5
        kf = StratifiedKFold(n_splits=k, random_state=0,shuffle=True)
        logging.info('SUCCESS!:Model trained and saved')
        return model, kf
    except BaseException:
        logging.info('ERROR!:Model not trained and not saved')

def train_and_compute_metrics(X_res, y_res,model,kf):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    try:
        accuracy = cross_val_score(model,  X_res,y_res, cv=kf, scoring='accuracy').mean()
        f1_macro = cross_val_score(model, X_res,y_res, cv=kf, scoring='f1_macro').mean()
        precision_macro = cross_val_score(model, X_res,y_res, cv=kf, scoring='precision_macro').mean()
        recall_macro = cross_val_score(model, X_res,y_res, cv=kf, scoring='recall_macro').mean()
        return accuracy, f1_macro, precision_macro,recall_macro
    except BaseException:
        logging.info('ERROR: Error occurred when scoring Models')


def inference(model, X_res):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model :
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds