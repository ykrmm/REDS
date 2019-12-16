import random,string,math,csv
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/ykarmim/Documents/Cours/Master/M2/REDS/src') # Mettre le path vers src sinon l'import fail.

from dataset import Dataset
from evaluation import Evaluation

from xgboost import XGBClassifier

if __name__ == "__main__":
    dataset = Dataset(drop_weight=True)
    higgs_df = dataset.get_df()
    Xtrain,Xtest, ytrain,ytest = dataset.split_train_test(fill_nan=False) # XgBoost prend en compte les valeurs NaN.
    weight_train,weight_test = dataset.get_weight_train_test()
    eval_ = Evaluation()

    model = XGBClassifier()
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    eval_.affichage_score("XGBoost",ytest,ypred,weights=weight_test)

