import random,string,math,csv
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/ykarmim/Documents/Cours/Master/M2/REDS/src') # Mettre le path vers src sinon l'import fail.

from dataset import Dataset
from evaluation import Evaluation

#import sklearn
#from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier


if __name__ == "__main__":
    dataset = Dataset(drop_weight=True)
    higgs_df = dataset.get_df()
    Xtrain,Xtest, ytrain,ytest = dataset.split_train_test(fill_nan=True)
    weight_train,weight_test = dataset.get_weight_train_test()
    eval_ = Evaluation()
    
    #Random forest
    score_min_rf = 3.0
    for bootstrap in [True,False]:
        for max_depth in range(3,10):
            for n_estimators in [10, 30, 50, 100, 200, 400, 600, 800, 1000]:
                rf = RandomForestClassifier(n_estimators=n_estimators,bootstrap=bootstrap,max_depth=max_depth)
                rf.fit(Xtrain,ytrain)
                ypred = rf.predict(Xtest)
                score = eval_.AMS(ytest,ypred,weights=weight_test)
                if score >= score_min_rf:
                    print(score)
                    bootstrap_opt = bootstrap
                    max_depth_opt = max_depth
                    n_estimators_opt = n_estimators 
                    score_min_rf = score 
                    
    #Bagging 
    score_min_bagg = 3.0
    for bootstrap in [True,False]:
        for n_estimators in range(10,100,10):
            bagging = BaggingClassifier(n_estimators=n_estimators,bootstrap=bootstrap)
            bagging.fit(Xtrain,ytrain)
            ypred = bagging.predict(Xtest)
            score = eval_.AMS(ytest,ypred,weights=weight_test)
            if score >= score_min_bagg:
                print(score)
                bootstrap_opt = bootstrap
                max_depth_opt = max_depth
                n_estimators_opt = n_estimators 
                score_min_bagg = score      
                
    #Adaboost
    score_min_ab = 3.0
    for n_estimators in [10, 30, 50, 100, 200, 400, 600, 800, 1000]:
        for learning_rate in range(0.1,1.1,0.1):
            adaboost = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
            adaboost.fit(Xtrain,ytrain)
            ypred = adaboost.predict(Xtest)
            score = eval_.AMS(ytest,ypred,weights=weight_test)
            if score >= score_min_ab:
                print(score)
                bootstrap_opt = bootstrap
                max_depth_opt = max_depth
                n_estimators_opt = n_estimators 
                score_min_ab = score
                
    #XGBoost 
    score_min_xg = 3.0
    #c koa ce truk
                    



