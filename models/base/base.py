import random,string,math,csv
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/ykarmim/Documents/Cours/Master/M2/REDS/src')

from dataset import Dataset
from evaluation import Evaluation


# Ce fichier contient nos baselines pour notre challenge.
# Nos modèles plus avancés se compareront à ces résultats. 
# Toutes nos méthodes/classes doivent renvoyer un ypred

def pure_random_model(X):
    
    return np.random.choice(['s','b'],len(X))

class Stratified:
    def __init__(self):
        pass
    def fit(self,Xtrain,ytrain):
        self.pb = ((y=='b').sum())/len(ytrain) # proba apparition b
        self.ps = ((y=='s').sum())/len(ytrain) # proba apparition s

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

    dataset = Dataset()
    higgs_df = dataset.__get_df__()
    X,y = dataset.__getX_Y__()
    Xtrain,Xtest, ytrain,ytest = dataset.split_train_test()




    #### Tests de nos différents modèles de bases  ####

    #random
    ypred = pure_random_model(Xtest)
    Evaluation.affichage_score("Random",ytest,ypred)

    #sratified
    stratified = Stratified()
    stratified.fit(Xtrain,ytrain)
    ypred = stratified.predict(Xtest,ytest)
    Evaluation.affichage_score("Stratified",ytest,ypred)

    #frequency 
    frequency = Frequency()
    frequency.fit(Xtrain,ytrain)
    ypred = frequency.predict(Xtest,ytest)
    Evaluation.affichage_score("Frequency",ytest,ypred)