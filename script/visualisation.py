import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import seaborn as sns
import sys
sys.path.insert(0, '"/Users/ykarmim/Documents/Cours/Master/M2/REDS/src')

from dataset import Dataset



dataset = Dataset("data/training.csv")
higgs_df = dataset.__get_df__()
X,y = dataset.__getX_Y__()
Xtrain,Xtest, ytrain,ytest = dataset.split_train_test()

# Visualisation classe déséquilibré 

#train['Label'].value_counts().plot(ax=ax, kind='bar')



# Analyse des données manquantes : 
cpt= 0 
for t in X : 
    if -999.0 in t : 
            cpt+=1

print("Pourcentage de lignes contenant des données manquantes : ",cpt/X.shape[0])

df_nan = higgs_df== -999.0

#count_nan = df_nan.sum(0)
#count_nan.plot.bar()