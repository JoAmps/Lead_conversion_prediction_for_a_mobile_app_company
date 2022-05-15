from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm
from lightgbm import LGBMClassifier
import wandb
from app.data_preprocess import preprocess_data
from load_and_clean_data import read_data


def main(name_model, model):

    wandb.init(project='lead_converts',
               group=name_model,  # Group experiments by model
               reinit=True
               )

    df = read_data('datasets/clean_leads_convert.csv')
    X_res, y_res,_ = preprocess_data(df, label='converted',training=True)

    k = 5
    kf = StratifiedKFold(n_splits=k, random_state=0, shuffle=True)

    accuracy = cross_val_score(
        model,
        X_res,
        y_res,
        cv=kf,
        scoring='accuracy').mean()
    f1_macro = cross_val_score(
        model,
        X_res,
        y_res,
        cv=kf,
        scoring='f1_macro').mean()
    precision_macro = cross_val_score(
        model, X_res, y_res, cv=kf, scoring='precision_macro').mean()
    recall_macro = cross_val_score(
        model, X_res, y_res, cv=kf, scoring='recall_macro').mean()

    wandb.log({'accuracy': accuracy,
               'f1_macro': f1_macro,
               'precision_macro': precision_macro,
               'recall_macro': recall_macro})


if __name__ == '__main__':
    models = {'LogisticRegression': LogisticRegression(solver='liblinear',
        max_iter=100000, random_state=0),
        'XGBClassifier': XGBClassifier(random_state=0),
              'DecisionTreeClassifier': DecisionTreeClassifier(),
              'RandomForestClassifier': RandomForestClassifier(random_state=0),
              'LGBMClassifier':LGBMClassifier(random_state=0)}

    for name, model in models.items():
        main(name, model)
