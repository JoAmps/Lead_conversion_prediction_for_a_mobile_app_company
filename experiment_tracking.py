from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import wandb
import pandas as pd
from functions.data_preprocess import process_data
from load_data import load_data


def main(name_model, model):

    wandb.init(project='leads', 
                group=name_model,  # Group experiments by model
                reinit=True   
    )

    # Load dataset
    df =load_data()
    X_res,y_res = process_data(df)

    k = 5
    kf = StratifiedKFold(n_splits=k, random_state=0,shuffle=True)

    accuracy = cross_val_score(model,  X_res,y_res, cv=kf, scoring='accuracy').mean()
    f1_macro = cross_val_score(model, X_res,y_res, cv=kf, scoring='f1_macro').mean()
    precision_macro = cross_val_score(model, X_res,y_res, cv=kf, scoring='precision_macro').mean()
    recall_macro = cross_val_score(model, X_res,y_res, cv=kf, scoring='recall_macro').mean()

    wandb.log({'accuracy': accuracy,
                'f1_macro': f1_macro,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro})

  

if __name__=='__main__':
    models = {'LogisticRegression': LogisticRegression(solver='liblinear',max_iter=10000,random_state=0),
            'LinearSVC': LinearSVC(),
            'DecisionTreeClassifier': DecisionTreeClassifier()}
    
    for name, model in models.items():
        main(name, model)  