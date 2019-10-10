import random,string,math,csv
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/ykarmim/Documents/Cours/Master/M2/REDS/src') # Insérer le path pour le dossier src

from dataset import Dataset
from evaluation import Evaluation
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
# Ce fichier contient nos baselines pour notre challenge.
# Nos modèles plus avancés se compareront à ces résultats. 
# Toutes nos méthodes/classes doivent renvoyer un ypred

def pure_random_model(X):
    
    return np.random.choice(['s','b'],len(X))

class Stratified:
    def __init__(self):
        pass
    def fit(self,Xtrain,ytrain):
        self.pb = ((ytrain=='b').sum())/len(ytrain) # proba apparition b
        self.ps = ((ytrain=='s').sum())/len(ytrain) # proba apparition s

    def predict(self,Xtest,ytest):
        return np.random.choice(['s','b'],len(Xtest),[self.ps,self.pb])

class Frequency:

    def __init__(self):
        pass

    def fit(self,Xtrain,ytrain):
        if (ytrain=='b').sum() > (ytrain=='s').sum():
            self.most_freq = 'b'
        else:
            self.most_freq = 's'
    
    def predict(self,Xtest,ytest):
        return np.random.choice([self.most_freq],len(ytest))

if __name__ == "__main__":

    
    dataset = Dataset(drop_weight=True)
    higgs_df = dataset.get_df()
    Xtrain,Xtest, ytrain,ytest = dataset.split_train_test(fill_nan=True)
    weight_train,weight_test = dataset.get_weight_train_test()

    eval_ = Evaluation()




    #### Tests de nos différents modèles de bases  ####
    
    #random
    ypred = pure_random_model(Xtest)
    eval_.affichage_score("Random",ytest,ypred,weights=weight_test)

    #sratified
    stratified = Stratified()
    stratified.fit(Xtrain,ytrain)
    ypred = stratified.predict(Xtest,ytest)
    eval_.affichage_score("Stratified",ytest,ypred,weights=weight_test)

    #frequency 
    frequency = Frequency()
    frequency.fit(Xtrain,ytrain)
    ypred = frequency.predict(Xtest,ytest)
    eval_.affichage_score("Frequency",ytest,ypred,weights=weight_test)

    """
    # Classifieur faible : Arbre de décision de profondeur 1 Weight = True

    tree_1 = DecisionTreeClassifier(random_state=0,max_depth=1)
    tree_1.fit(Xtrain,ytrain)
    ypred = tree_1.predict(Xtest)
    Evaluation.affichage_score("Decision Tree Depth 1 Weight = True",ytest,ypred)
    


    dataset = Dataset(drop_weight=True)
    higgs_df = dataset.__get_df__()
    X,y = dataset.__getX_Y__()
    Xtrain,Xtest, ytrain,ytest = dataset.split_train_test()"""
    
    # Arbre de decision de profondeur 1 Weight = False
    
    
    tree_1 = DecisionTreeClassifier(random_state=0,max_depth=1)
    tree_1.fit(Xtrain,ytrain)
    ypred = tree_1.predict(Xtest)
    eval_.affichage_score("Decision Tree Depth 1 Weight = False",ytest,ypred,weights=weight_test)


     # Arbre de decision de profondeur 2 Weight = False
    tree_1 = DecisionTreeClassifier(random_state=0,max_depth=2)
    tree_1.fit(Xtrain,ytrain)
    ypred = tree_1.predict(Xtest)
    eval_.affichage_score("Decision Tree Depth 2 Weight = False",ytest,ypred,weights=weight_test)

    # Naive Bayes Weight = False

    gnb = GaussianNB()
    ypred1 = gnb.fit(Xtrain, ytrain).predict(Xtest)
    eval_.affichage_score("Naive Bayes",ytest,ypred1,weights=weight_test)

    # Perceptron Weight = False

    perceptron = Perceptron(tol=1e-3, random_state=0)
    perceptron.fit(Xtrain,ytrain)
    ypred_perceptron = perceptron.predict(Xtest)
    eval_.affichage_score("Perceptron",ytest,ypred_perceptron,weights=weight_test)


    # SVM Linear SVC Weight = False
    svc = LinearSVC(random_state=0, tol=1e-5)
    svc.fit(Xtrain,ytrain)
    ypred_svc = svc.predict(Xtest)   
    eval_.affichage_score("SVM Linear SVC",ytest,ypred_svc,weights=weight_test)
