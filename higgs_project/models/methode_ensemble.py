import random,string,math,csv
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/ykarmim/Documents/Cours/Master/M2/REDS/src') # Mettre le path vers src sinon l'import fail.

from dataset import Dataset
from evaluation import Evaluation
from xgboost import XGBClassifier
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import scale

import ast











if __name__ == "__main__":
    dataset = Dataset(drop_weight=True)
    higgs_df = dataset.get_df()
    Xtrain,Xtest, ytrain,ytest = dataset.split_train_test(fill_nan=True)
    weight_train,weight_test = dataset.get_weight_train_test()
    eval_ = Evaluation()
    Xtrain = scale(Xtrain)
    Xtest = scale(Xtest)

    """
    #Decision tree automatic depth

    tree_1 = DecisionTreeClassifier(random_state=0)
    tree_1.fit(Xtrain,ytrain)
    ypred = tree_1.predict(Xtest)
    ypred_train = tree_1.predict(Xtrain)
    eval_.affichage_score("Decision Tree automatic depth Weight = False, Test",ytest,ypred,weights=weight_test)
    eval_.affichage_score("Decision Tree automatic depth Weight = False, Train",ytrain,ypred_train,weights=weight_train)
    


    

     # Decision tree fixed depth 5
    tree_1 = DecisionTreeClassifier(random_state=0,max_depth=5)
    tree_1.fit(Xtrain,ytrain)
    ypred = tree_1.predict(Xtest)
    ypred_train = tree_1.predict(Xtrain)
    eval_.affichage_score("Decision Tree  depth 5 Weight = False, Test",ytest,ypred,weights=weight_test)
    eval_.affichage_score("Decision Tree  depth 5 Weight = False, Train",ytrain,ypred_train,weights=weight_train)
    
    # à faire : cross val pour trouver le meilleur hyper paramètre pour la profondeur maximal de l'arbre 

   """
    # Random Forest Bagging bootstrap = True , n_estimator = 100
    optim_param= pd.read_csv('results/optimal_parameters_rf.csv')
    params = optim_param['params']
    params = ast.literal_eval(params[0])
    rf = RandomForestClassifier(**params)
    rf.fit(Xtrain,ytrain)
    ypred = rf.predict(scale(Xtest))
    eval_.affichage_score("Random Forest bootstrap = True ",ytest,ypred,weights=weight_test)

    """
    rf = RandomForestClassifier(n_estimators=100,bootstrap=False,max_depth=5)
    rf.fit(Xtrain,ytrain)
    ypred = rf.predict(Xtest)
    eval_.affichage_score("Random Forest bootstrap = False ",ytest,ypred,weights=weight_test)
     # à faire cross val pour les hyper-paramètres
    """
     # Extra Trees 

    """Extra trees classifier.
       This classifier is very similar to a random forest; in extremely
       randomized trees, randomness goes one step further in the way splits are
       computed. As in random forests, a random subset of candidate features is
       used, but instead of looking for the most discriminative thresholds,
       thresholds are drawn at random for each candidate feature and the best of
       these randomly-generated thresholds is picked as the splitting rule.

       This usually allows to reduce the variance of the model a bit more, at
       the expense of a slightly greater increase in bias:
    """
    """
    # bootstrap = False
    et = ExtraTreesClassifier(n_estimators=100,max_depth=5,bootstrap=False)
    et.fit(Xtrain,ytrain)
    ypred = et.predict(Xtest)
    eval_.affichage_score("Extra Trees bootstrap = False ",ytest,ypred,weights=weight_test)

    # bootstrap = True
    et = ExtraTreesClassifier(n_estimators=100,max_depth=5,bootstrap=True)
    et.fit(Xtrain,ytrain)
    ypred = et.predict(Xtest)
    eval_.affichage_score("Extra Trees bootstrap = True ",ytest,ypred,weights=weight_test)

    # à faire cross val pour les hyper paramètres
    """
    # Bagging 
    optim_param= pd.read_csv('results/optimal_parameters_bagging.csv')
    params = optim_param['params']
    params = ast.literal_eval(params[0])
    bagging = BaggingClassifier(**params)
    bagging.fit(Xtrain,ytrain)
    ypred = bagging.predict(Xtest)
    eval_.affichage_score("Bagging n_estimators = 10 ",ytest,ypred,weights=weight_test)

    """
    bagging1 = BaggingClassifier(n_estimators=100)
    bagging1.fit(Xtrain,ytrain)
    ypred = bagging1.predict(Xtest)
    eval_.affichage_score("Bagging n_estimators = 100 ",ytest,ypred,weights=weight_test)
    """

    # AdaBoost
    optim_param= pd.read_csv('results/optimal_parameters_adaboost.csv')
    params = optim_param['params']
    params = ast.literal_eval(params[0])
    adaboost = AdaBoostClassifier(**params)
    adaboost.fit(Xtrain,ytrain)
    ypred = adaboost.predict(Xtest)
    eval_.affichage_score("Adaboost ",ytest,ypred,weights=weight_test)
    
    #XGboost 
    optim_param= pd.read_csv('results/optimal_parameters_adaboost.csv')
    params = optim_param['params']
    params = ast.literal_eval(params[0])
    model = XGBClassifier(**params)
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    eval_.affichage_score("XGBoost",ytest,ypred,weights=weight_test)