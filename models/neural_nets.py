import random,string,math,csv
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/ykarmim/Documents/Cours/Master/M2/REDS/src') # Mettre le path vers src sinon l'import fail.

from dataset import Dataset
from evaluation import Evaluation
import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler # utile pour split train test 
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from sklearn.preprocessing import scale

import ast




class Higgs_Dataset(data.Dataset):
    def __init__(self,X,Y):
        Y = np.where(Y=='s', 1, Y)
        Y = np.where(Y=='b', 0, Y) 
        Y = Y.astype(int)
        X = X.astype(float) 
        self.data = torch.from_numpy(X).float()
        self.labels = torch.from_numpy(Y).float()
        
    def __getitem__(self,index):
        return self.data[index],self.labels[index]

    def __len__(self):
        return len(self.labels)
    
    def get_prediction(self,y):
        y = y.data.numpy()
        y = np.where(y==1,'s',y)
        y = np.where(y==0,'b',y)
        return y 

class MLP_Higgs(torch.nn.Module):
    def __init__(self,D_in,H,H2,D_out):
        super(MLP_Higgs,self).__init__()
        self.linear1 = torch.nn.Linear(D_in,H)
        self.linear2 = torch.nn.Linear(H,H2)
        self.linear3 = torch.nn.Linear(H2,H2)
        self.linear4 = torch.nn.Linear(H2,H2//2)
        self.linear5 = torch.nn.Linear(H2//2,D_out)
        self.activ1 = torch.nn.Tanh()
        self.sortie = torch.nn.Sigmoid()
        

    def forward(self,x):
        y = self.linear1(x)
        y = self.activ1(y)
        y = self.linear2(y)#.squeeze()
        y = self.activ1(y)
        y = self.linear3(y)
        y = self.activ1(y)
        y = self.linear4(y)
        y = self.activ1(y)
        y = self.linear5(y)
        y = self.sortie(y)
        return y 






if __name__ == "__main__":
    dataset = Dataset(drop_weight=True)
    higgs_df = dataset.get_df()
    keep_columns=['DER_mass_transverse_met_lep',\
         'DER_mass_MMC', 'DER_met_phi_centrality','DER_mass_vis',\
              'DER_deltar_tau_lep', 'PRI_tau_pt','DER_pt_ratio_lep_tau', 'PRI_met', 'DER_sum_pt']
    dataset.columns_to_keep(keep_columns)
    Xtrain,Xtest, ytrain,ytest = dataset.split_train_test(fill_nan=True)
    weight_train,weight_test = dataset.get_weight_train_test()
    eval_ = Evaluation()
    Xtrain = scale(Xtrain)
    Xtest = scale(Xtest)



    batch_size = 20
    shuffle_dataset = True
    random_seed = 42

    n_epoch = 100
    learning_rate = 10e-3
    dataset_train = Higgs_Dataset(X=Xtrain,Y=ytrain)
    train_loader = data.DataLoader(dataset_train,shuffle=shuffle_dataset,batch_size=batch_size) 

    dataset_test = Higgs_Dataset(X=Xtest,Y=ytest)
    test_loader = data.DataLoader(dataset_test,shuffle=shuffle_dataset,batch_size=batch_size) 

    # Script entrainement model perceval 

    model = MLP_Higgs(D_in=30,H=20,H2=10,D_out=1)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    writer = SummaryWriter()


    print(" ------------ ENTRAINEMENT RESEAU DE NEURONES ---------------")
    for ep in range(n_epoch):
        print("EPOCHS : ",ep)
        for i, (x, y) in enumerate(train_loader):
            model.train()
            y = y#.float()
            x = x#.double()
            
            pred = model(x)#.double()
            print(pred)
            loss = criterion(pred, y)
            # print('loss', loss ," Prédiction : ", pred, "y True : ",y)
            writer.add_scalar('Loss/train', loss, ep)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        for i,(x,y) in enumerate(test_loader):
            with torch.no_grad():
                model.eval()
                pred = model(x)
                loss = criterion(pred,y)
                # print('loss', loss ," Prédiction : ", pred, "y True : ",y)
                writer.add_scalar('Loss/validation', loss, ep)


    ypred = dataset_train.get_prediction(ynn)
    eval_.affichage_score("Neural Network ",ytest,ypred,weights=weight_test)