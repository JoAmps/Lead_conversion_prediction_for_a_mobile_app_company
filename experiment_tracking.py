from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
import wandb
import pandas as pd
from functions.data_preprocess import preprocess_data
from load_and_clean_data import process_data


def main(name_model, model):

    wandb.init(project='lead_conversions', 
                group=name_model,  # Group experiments by model
                reinit=True   
    )

    df =process_data()
    X_res,y_res = preprocess_data(df)

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
    models = {'LogisticRegression': LogisticRegression(solver='liblinear',max_iter=100000,random_state=0),
            'LinearSVC': LinearSVC(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'AdaBoostClassifier':AdaBoostClassifier()}
    
    for name, model in models.items():
        main(name, model)  