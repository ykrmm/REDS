import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import seaborn as sns
from sklearn.model_selection import train_test_split





class Dataset:
        def __init__(self,file_path="../data/cern/atlas-higgs-challenge-2014-v2.csv",drop_weight=True):
                """
                drop_weight : Booléen pour supprimer l'attribut Weight de nos données d'apprentissage  et de test.
                """
                self.higgs_df =  pd.read_csv(file_path)
                self.higgs_df = self.higgs_df.drop(['EventId'],axis=1)
                self.higgs_df= self.higgs_df.replace(-999.0,np.nan) #On remplace les valeurs manquantes par NaN.
                self.drop_weight = drop_weight
                self.split_done = False
                if self.drop_weight:
                        self.higgs_df = self.higgs_df.drop(['Weight'],axis=1) # On a besoin de conserver les poids pour l'évaluation AMS.
                
        
        def columns_to_keep(self,columns_to_keep):
                for col in self.higgs_df.columns:
                        if col not in columns_to_keep:
                                self.higgs_df = self.higgs_df.drop([col],axis=1)
        def drop_column_df(self,columns_to_drop):
                # Call this function before split_train_test. So it will delete columns for all dataset.
                self.higgs_df = self.higgs_df.drop(columns_to_drop,axis=1)
                return self.higgs_df
        
        def get_df(self):
                return self.higgs_df
        
        def get_df_train_test_pub(self):
                """
                        return when split_train_test done the dataframe of higgs_train,higgs_test and higgs_public.
                """
                if self.split_done:
                        return  self.higgs_df_train,self.higgs_df_private,self.higgs_df_public
        
        def get_weight_array(self):

                return self.weight_array

        def get_weight_train_test(self):
                if self.split_done:
                        return self.weight_train, self.weight_test
                else:
                        print("call split_train_test() first.")

        def get_X_Y_Weight_test_public(self):
                if self.split_done : 
                        return self.Xtest_public, self.Ytest_public
                else:
                        print("call split_train_test() first")
        
        def _fill_nan_values(self):
                """
                        Private method.
                """
                mean_train = self.higgs_df_train.mean()
                self.higgs_df_train.fillna(mean_train,inplace=True)
                self.higgs_df_public.fillna(mean_train,inplace=True)
                self.higgs_df_private.fillna(mean_train,inplace=True)

        
        def split_train_test(self,fill_nan=True):
                """
                        Split the dataset into 3 datasets : train, test and test public. 
                        fill_nan : Bool -> Replace NaN by the mean average of train's columns.
                """

                self.higgs_df_train = self.higgs_df.loc[self.higgs_df['KaggleSet'] == 't']
                self.higgs_df_public = self.higgs_df.loc[self.higgs_df['KaggleSet'] == 'b']
                self.higgs_df_private = self.higgs_df.loc[self.higgs_df['KaggleSet'] == 'v']
                
                try:
                        self.higgs_df= self.higgs_df.drop(['KaggleSet'],axis=1)
                        self.higgs_df_train= self.higgs_df_train.drop(['KaggleSet'],axis=1)
                        self.higgs_df_public= self.higgs_df_public.drop(['KaggleSet'],axis=1)
                        self.higgs_df_private= self.higgs_df_private.drop(['KaggleSet'],axis=1)
                except:
                        print("Les colonnes kaggleset n'ont pas pu être supprimés")
                if fill_nan:
                        self._fill_nan_values() # Replace nan value by the mean average of train's columns.

                self.weight_train = self.higgs_df_train['KaggleWeight'].values
                self.weight_test = self.higgs_df_private['KaggleWeight'].values
                self.weight_test_public = self.higgs_df_public['KaggleWeight'].values
                self.higgs_df_train= self.higgs_df_train.drop(['KaggleWeight'],axis=1)
                self.higgs_df_public= self.higgs_df_public.drop(['KaggleWeight'],axis=1)
                self.higgs_df_private= self.higgs_df_private.drop(['KaggleWeight'],axis=1)
                

                self.Xtrain = self.higgs_df_train.values
                self.Ytrain = self.higgs_df_train['Label'].values
                self.Xtrain = np.delete(self.Xtrain,len(self.Xtrain[0])-1,axis=1) # supprime les labels
                
                self.Xtest = self.higgs_df_private.values
                self.Ytest = self.higgs_df_private['Label'].values
                self.Xtest = np.delete(self.Xtest,len(self.Xtest[0])-1,axis=1) # supprime les labels
                
                self.Xtest_public = self.higgs_df_public.values
                self.Ytest_public = self.higgs_df_public['Label'].values
                self.Xtest_public = np.delete(self.Xtest_public,len(self.Xtest_public[0])-1,axis=1) # supprime les labels

                self.split_done = True
                
                return self.Xtrain,self.Xtest,self.Ytrain,self.Ytest









if __name__ == "__main__":

        dataset = Dataset(drop_weight=True)
        higgs_df = dataset.get_df()
        Xtrain,Xtest, ytrain,ytest = dataset.split_train_test(fill_nan=True)
        weight_train,weight_test = dataset.get_weight_train_test()
