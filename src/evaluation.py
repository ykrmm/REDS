from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import numpy as np

"""
    Class qui nous permet d'évaluer nos modèles et de réaliser des affichages.
"""

class Evaluation:
    def __init__(self):
        pass 

    @staticmethod
    def accuracy(ytrue,ypred):
        return accuracy_score(ytrue,ypred)

    @staticmethod
    def precision(ytrue,ypred):
        return precision_score(ytrue,ypred) 
    
    @staticmethod
    def rappel(ytrue,ypred):
        return recall_score(ytrue,ypred)

    @staticmethod
    def f1(ytrue,ypred):
        return f1_score(ytrue,ypred)

    
    def AMS(self,ytrue,ypred,weights):
        s=0
        b=0
        for i in range(len(ytrue)):
            if (ypred[i]=='s'):
                if (ytrue=='b'):
                    b+=weights[i]
                else :
                    s+=weights[i]
        AMS = np.sqrt(2*((s+b+10)*np.log(1+(s/(b+10)))-s))
        return AMS

    @staticmethod
    def affichage_score(model_name,ytrue,ypred,weights=None,prec=4):
        """
            model_name : nom du modèle utilisé pour l'affichage
        """
        print("\n\n------------------ Evaluation {} ------------------\n\n".format(model_name))
        print("|xxxxxxxxxxxxx|   ACCURACY  = {:.{prec}f}    |xxxxxxxxxxxx|".format(accuracy_score(ytrue,ypred),prec=prec))
        print("|xxxxxxxxxxxxx|   RECALL    = {:.{prec}f}    |xxxxxxxxxxxx|".format(recall_score(ytrue,ypred,pos_label="s"),prec=prec))
        print("|xxxxxxxxxxxxx|   PRECISION = {:.{prec}f}    |xxxxxxxxxxxx|".format(precision_score(ytrue,ypred,pos_label="s"),prec=prec))
        print("|xxxxxxxxxxxxx|   SCORE F1  = {:.{prec}f}    |xxxxxxxxxxxx|".format(f1_score(ytrue,ypred,pos_label="s"),prec=prec))
        #print("|xxxxxxxxxxxxx|   AMS SCORE = {:.{prec}f}    |xxxxxxxxxxxx|\n\n".format(self.AMS(ytrue, ypred,weights),prec=prec))