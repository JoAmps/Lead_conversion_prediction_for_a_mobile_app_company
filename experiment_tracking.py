from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import wandb
import pandas as pd
from data_preprocess import process_data


def main(name_model, model):

    wandb.init(project='leads', 
                group=name_model, # Group experiments by model
    )

    # Load dataset
    df = pd.read_csv('datasets/lead_convert.csv',index_col=0)
    X_res,y_res = process_data(df)

    # Split into train and test set
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    #result = cross_val_score(cv_xgb.best_estimator_, X_res,y_res, cv = kf, scoring='precision_macro',n_jobs=-1)
    k = 5
    kf = StratifiedKFold(n_splits=k, random_state=0,shuffle=True)

    accuracy = cross_val_score(model,  X_res,y_res, cv=kf, scoring='accuracy',n_jobs=-1).mean()
    f1_macro = cross_val_score(model, X_res,y_res, cv=kf, scoring='f1_macro',n_jobs=-1).mean()
    precision_macro = cross_val_score(model, X_res,y_res, cv=kf, scoring='precision_macro',n_jobs=-1).mean()
    recall_macro = cross_val_score(model, X_res,y_res, cv=kf, scoring='recall_macro',n_jobs=-1).mean()

    wandb.log({'accuracy': accuracy,
                'f1_macro': f1_macro,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro})

  

if __name__=='__main__':
    models = {'LogisticRegression': LogisticRegression(solver='liblinear',max_iter=1000),
            'LinearSVC': LinearSVC(),
            'DecisionTreeClassifier': DecisionTreeClassifier()}
    
    for name, model in models.items():
        main(name, model)  