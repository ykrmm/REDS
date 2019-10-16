import random,string,math,csv
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/ykarmim/Documents/Cours/Master/M2/REDS/src') # Mettre le path vers src sinon l'import fail.

from dataset import Dataset
from evaluation import Evaluation

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score 

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
    all_result_df = pd.DataFrame(columns=['models','params','AMS','Accuracy','Precision','Recall']) # Notre tableau de résultat
    optimal_parameters = pd.DataFrame(columns=['models','params','AMS']) # Notre tableau résultats optimaux
    
    #Random forest
    score_opt_ams = 0
    for bootstrap in [True,False]:
        for max_depth in range(3,10):
            for n_estimators in [10, 30, 50, 100, 200, 400, 600, 800, 1000]:

                rf = RandomForestClassifier(n_estimators=n_estimators,bootstrap=bootstrap,max_depth=max_depth)
                rf.fit(Xtrain,ytrain)
                ypred = rf.predict(Xtest)
                score_ams = eval_.AMS(ytest,ypred,weights=weight_test)
                score_accuracy = eval_.accuracy(ytest,ypred)
                score_precision = eval_.precision(ytest,ypred)
                score_recall = eval_.recall(ytest,ypred)
                l = {'models':'Random Forest','params':str(rf.get_params()),\
                    'AMS':score_ams,'Accuracy':score_accuracy,'Precision':score_precision,'Recall':score_recall}
                all_result_df=all_result_df.append(l,ignore_index=True)
                if score_ams > score_opt_ams:
                    optim_param = str(rf.get_params())
                    score_opt_ams = score_ams

    l = {'models':'Random Forest','params':optim_param,'AMS':score_opt_ams}
    optimal_parameters=optimal_parameters.append(l)          
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

    dataset = Dataset(drop_weight=True)
    higgs_df = dataset.get_df()
    Xtrain,Xtest, ytrain,ytest = dataset.split_train_test(fill_nan=True)
    weight_train,weight_test = dataset.get_weight_train_test()
    eval_ = Evaluation() 

    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
    #XGBoost 
    score_min_xg = 3.0
    #c koa ce truk
                    



