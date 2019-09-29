import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import seaborn as sns
from sklearn.model_selection import train_test_split





class Dataset:
        def __init__(self,file_path="/Users/ykarmim/Documents/Cours/Master/M2/REDS/data/training.csv",drop_weight=True):
                """
                drop_weight : Booléen pour supprimer l'attribut Weight de nos données.
                """
                self.higgs_df =  pd.read_csv(file_path)
                if drop_weight:
                        self.higgs_df= self.higgs_df.drop(['Weight'],axis=1)
                self.is_X_y_build = False
        
        def __getX_Y__(self):

                self.X = np.delete(self.higgs_df.values, 0, axis=1)
                self.X = np.delete(self.X,len(self.X[0])-1,axis=1)
                self.Y = self.higgs_df['Label'].values
                self.is_X_y_build = True
                return self.X ,self.Y

        def drop_column_df(self,columns_to_drop):

                self.higgs_df = self.higgs_df.drop(columns_to_drop,axis=1)
                return self.higgs_df
        
        def __get_df__(self):

                return self.higgs_df

        
        def split_train_test(self,test_size=0.33,random_state=42):

                if( not self.is_X_y_build):
                        print("You  have to call first __getX_Y() before to split in train test.\n")

                else:
                        self.Xtrain,self.Xtest,self.Ytrain,self.Ytest = train_test_split\
                                (self.X, self.Y, test_size=test_size, random_state=random_state)
                
                return self.Xtrain,self.Xtest,self.Ytrain,self.Ytest 









if __name__ == "__main__":

        dataset = Dataset()