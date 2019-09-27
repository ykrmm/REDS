from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


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

    @staticmethod
    def affichage_score(model_name,ytrue,ypred,prec=4):
        """
            model_name : nom du modèle utilisé pour l'affichage
        """
        print("\n\n------------------ Evaluation {} ------------------\n\n".format(model_name))
        print("|xxxxxxxxxxxxx|   ACCURACY  = {:.{prec}f}    |xxxxxxxxxxxx|".format(accuracy_score(ytrue,ypred),prec=prec))
        print("|xxxxxxxxxxxxx|   RECALL    = {:.{prec}f}    |xxxxxxxxxxxx|".format(recall_score(ytrue,ypred,pos_label="s"),prec=prec))
        print("|xxxxxxxxxxxxx|   PRECISION = {:.{prec}f}    |xxxxxxxxxxxx|".format(precision_score(ytrue,ypred,pos_label="s"),prec=prec))
        print("|xxxxxxxxxxxxx|   SCORE F1  = {:.{prec}f}    |xxxxxxxxxxxx|\n\n".format(f1_score(ytrue,ypred,pos_label="s"),prec=prec))
        print("-----------------------------------------------------------")