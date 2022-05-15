from sklearn.metrics import f1_score, precision_score, recall_score,\
 accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np



def split_data(data):
    """
    Splits the data into training and testing
    Inputs
    -------
    data : pandas dataframe
           The cleaned data
    Returns
    -------
    train
        train data for training
    test
        test data for validation
    """
    try:
        train, test = train_test_split(
            data, test_size=0.1, random_state=0, stratify=data['converted'])
        logging.info('SUCCESS!:Data split successfully')
        return train, test
    except BaseException:
        logging.info('Error!:Error whiles splitting data')


def train_model(X_train, y_train):
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
        model = RandomForestClassifier(random_state=0)
        model.fit(X_train, y_train)
        logging.info('SUCCESS!:Model trained and saved')
        return model
    except BaseException:
        logging.info('ERROR!:Model not trained and not saved')


def model_predictions(X_test, model):
    """
    Performs prediction on the independent testing data using the trained
    machine learning model
    Inputs
    ------
    X_test : np.array
             Testing data
    y_test : np.array
             Test labels
    Returns
    -------
    predictions : int
    """
    try:
        predictions = model.predict(X_test)
        logging.info('SUCCESS!:Model predictions generated')
        return predictions
    except BaseException:
        logging.info('ERROR!:Model predictions not generated')


def compute_metrics(y, predictions):
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
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        logging.info('SUCCESS: Model scoring completed')
        return accuracy, precision, recall, f1
    except BaseException:
        logging.info('ERROR: Error occurred when scoring Models')


def plot_visualizations(predictions, y_test,model,X_test):
    try:
        # confusion matrix
        sns.heatmap(
            confusion_matrix(
                predictions,
                y_test),
            annot=True,
            annot_kws={
                "fontsize": 20},
            fmt='d',
            cbar=False,
            cmap='icefire')
        plt.title('Confusion Matrix', color='navy', fontsize=15)
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.savefig("plots/confusionmatrix.png", bbox_inches='tight', dpi=1000)
        # roc curve
        fpr, tpr, _ = roc_curve(y_test, predictions)
        plt.figure(figsize=(7, 7))
        plt.plot(
            fpr,
            tpr,
            label='AUC (Area = %0.2f)' %
            roc_auc_score(
                y_test,
                predictions))
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Sensitivity : False Positive Ratio')
        plt.ylabel('True Positive Ratio')
        plt.title('ROC and AUC of the XGboost Model')
        plt.legend()
        plt.savefig("plots/roc_curve.png", bbox_inches='tight', dpi=1000)
        
        #generate feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [X_test.columns[i] for i in indices]
        plt.figure(figsize=(20,5))
        plt.title("Feature Importance")
        plt.ylabel('Importance') 
        plt.bar(range(X_test.shape[1]), importances[indices])
        plt.xticks(range(X_test.shape[1]), names, rotation=90);
        plt.savefig("plots/feature_importance.png", bbox_inches='tight', dpi=1000)
        logging.info('SUCCESS!:Visualizations plotted and saved!')
    except BaseException:
        logging.info('ERROR: Visualizations could not be plotted and saved!')



def inference(model, X):
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
