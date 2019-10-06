import random,string,math,csv
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/ykarmim/Documents/Cours/Master/M2/REDS/src') # Mettre le path vers src sinon l'import fail.

from dataset import Dataset
from evaluation import Evaluation

import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import HistGradientBoostingRegressor
















if __name__ == "__main__":
    dataset = Dataset(drop_weight=True)
    higgs_df = dataset.__get_df__()
    X,y = dataset.__getX_Y__()
    Xtrain,Xtest, ytrain,ytest = dataset.split_train_test()

    """
    #Decision tree automatic depth

    tree_1 = DecisionTreeClassifier(random_state=0)
    tree_1.fit(Xtrain,ytrain)
    ypred = tree_1.predict(Xtest)
    ypred_train = tree_1.predict(Xtrain)
    Evaluation.affichage_score("Decision Tree automatic depth Weight = False, Test",ytest,ypred)
    Evaluation.affichage_score("Decision Tree automatic depth Weight = False, Train",ytrain,ypred_train)
    


    

     # Decision tree fixed depth 5
    tree_1 = DecisionTreeClassifier(random_state=0,max_depth=5)
    tree_1.fit(Xtrain,ytrain)
    ypred = tree_1.predict(Xtest)
    ypred_train = tree_1.predict(Xtrain)
    Evaluation.affichage_score("Decision Tree  depth 5 Weight = False, Test",ytest,ypred)
    Evaluation.affichage_score("Decision Tree  depth 5 Weight = False, Train",ytrain,ypred_train)
    """
    # à faire : cross val pour trouver le meilleur hyper paramètre pour la profondeur maximal de l'arbre 


    # Random Forest Bagging bootstrap = True , n_estimator = 100

    rf = RandomForestClassifier(n_estimators=100,bootstrap=True,max_depth=5)
    rf.fit(Xtrain,ytrain)
    ypred = rf.predict(Xtest)
    Evaluation.affichage_score("Random Forest bootstrap = True ",ytest,ypred)

    rf = RandomForestClassifier(n_estimators=100,bootstrap=False,max_depth=5)
    rf.fit(Xtrain,ytrain)
    ypred = rf.predict(Xtest)
    Evaluation.affichage_score("Random Forest bootstrap = False ",ytest,ypred)
     # à faire cross val pour les hyper-paramètres

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

    # bootstrap = False
    et = ExtraTreesClassifier(n_estimators=100,max_depth=5,bootstrap=False)
    et.fit(Xtrain,ytrain)
    ypred = et.predict(Xtest)
    Evaluation.affichage_score("Extra Trees bootstrap = False ",ytest,ypred)

    # bootstrap = True
    et = ExtraTreesClassifier(n_estimators=100,max_depth=5,bootstrap=True)
    et.fit(Xtrain,ytrain)
    ypred = et.predict(Xtest)
    Evaluation.affichage_score("Extra Trees bootstrap = True ",ytest,ypred)

    # à faire cross val pour les hyper paramètres

    # Bagging 

    bagging = BaggingClassifier(n_estimators=10)
    bagging.fit(Xtrain,ytrain)
    ypred = bagging.predict(Xtest)
    Evaluation.affichage_score("Bagging n_estimators = 10 ",ytest,ypred)

    bagging1 = BaggingClassifier(n_estimators=100)
    bagging1.fit(Xtrain,ytrain)
    ypred = bagging1.predict(Xtest)
    Evaluation.affichage_score("Bagging n_estimators = 100 ",ytest,ypred)


    # AdaBoost
    adaboost = AdaBoostClassifier()
    adaboost.fit(Xtrain,ytrain)
    ypred = adaboost.predict(Xtest)
    Evaluation.affichage_score("Adaboost ",ytest,ypred)
