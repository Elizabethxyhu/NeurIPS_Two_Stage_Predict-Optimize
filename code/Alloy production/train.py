import sys
import numpy as np
import random
import pandas as pd
import math, time
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import gurobipy as gp
import logging
import copy
from collections import defaultdict
import joblib
import gurobipy as gp
from gurobipy import GRB
import random

rowSizeG = 2
colSizeG = 10
varNum = colSizeG
featureNum = 4096
trainSize = 350
cap = [627.54, 369.72]
#cap = [0.8, 60, 40, 2.5]
penaltyTerm = 0.25

def get_xTrue(cTemp, GTemp, hTemp, n_instance):
#    print(cTemp, hTemp)
    obj_list = []
    for num in range(n_instance):
        c = np.zeros((colSizeG))
        cntC = num * colSizeG
        for i in range(colSizeG):
            c[i] = cTemp[cntC]
            cntC = cntC + 1
        c = c.tolist()
        h = np.zeros((rowSizeG))
        cntH = num * rowSizeG
        for i in range(rowSizeG):
            h[i] = hTemp[cntH]
            cntH = cntH + 1
        h = h.tolist()
        
        G = np.zeros((rowSizeG, colSizeG))
        cnt = num * rowSizeG * colSizeG
        for i in range(rowSizeG):
            for j in range(colSizeG):
                G[i][j] = GTemp[cnt]
                cnt = cnt + 1
        G = G.tolist()
        
#        print(c, h)
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(varNum, vtype=GRB.CONTINUOUS, name='x')
        m.setObjective(x.prod(c), GRB.MINIMIZE)
        for i in range(rowSizeG):
            m.addConstr((x.prod(G[i])) >= h[i])

        m.optimize()
        sol = np.zeros(varNum)
        for i in range(varNum):
            sol[i] = x[i].x
        
        objective = m.objVal
        obj_list.append(objective)
#        print(sol, objective)
        
    return sol

def actual_obj(cTemp, GTemp, hTemp, n_instance):
#    print(cTemp, hTemp)
    obj_list = []
    for num in range(n_instance):
        c = np.zeros((colSizeG))
        cntC = num * colSizeG
        for i in range(colSizeG):
            c[i] = cTemp[cntC]
            cntC = cntC + 1
        c = c.tolist()
        h = np.zeros((rowSizeG))
        cntH = num * rowSizeG
        for i in range(rowSizeG):
            h[i] = hTemp[cntH]
            cntH = cntH + 1
        h = h.tolist()
        
        G = np.zeros((rowSizeG, colSizeG))
        cnt = num * rowSizeG * colSizeG
        for i in range(rowSizeG):
            for j in range(colSizeG):
                G[i][j] = GTemp[cnt]
                cnt = cnt + 1
        G = G.tolist()
        
#        print(c, h)
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(varNum, vtype=GRB.CONTINUOUS, name='x')
        m.setObjective(x.prod(c), GRB.MINIMIZE)
        for i in range(rowSizeG):
            m.addConstr((x.prod(G[i])) >= h[i])

        m.optimize()
        sol = []
        for i in range(varNum):
            sol.append(x[i].x)
        
        objective = m.objVal
        obj_list.append(objective)
#        print(sol, objective)
        
    return np.array(obj_list)
    
def correction_single_obj(c, realG, preG, h, penalty):
#    print(c, realG, preG, h)
    rowSizeG = realG.shape[0]
    if preG.all() >= 0:
        realG = realG.tolist()
        preG = preG.tolist()
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(varNum, vtype=GRB.CONTINUOUS, name='x')
        m.setObjective(x.prod(c), GRB.MINIMIZE)
        for i in range(rowSizeG):
            m.addConstr((x.prod(preG[i])) >= h[i])

        m.optimize()
        predSol = []
    #    print(x)
        try:
            for i in range(varNum):
                predSol.append(x[i].x)
    #        print(sol)
        #        for i in range(allPathNum):
        #            if sol[i] != 0:
        #                print(i, end=" ")
            objective = m.objVal
    #        print(predSol, objective)
        except:
            for i in range(varNum):
                predSol.append(0)
            objective = 0
        
        # Stage 2:
        m2 = gp.Model()
        m2.setParam('OutputFlag', 0)
        x = m2.addVars(varNum, vtype=GRB.CONTINUOUS, name='x')
        sigma = m2.addVars(varNum, vtype=GRB.CONTINUOUS, name='sigma')

        OBJ = objective
        for i in range(varNum):
            OBJ = OBJ + (1 + penalty[i]) * c[i] * sigma[i]
        m2.setObjective(OBJ, GRB.MINIMIZE)

        for i in range(rowSizeG):
            m2.addConstr((x.prod(realG[i]) + sigma.prod(realG[i])) >= h[i])
        for i in range(varNum):
            m2.addConstr(x[i] == predSol[i])

        m2.optimize()
        objective = m2.objVal
        sol = []
        for i in range(varNum):
            sol.append(sigma[i].x)
#        print(sol, objective)
        
    return objective
    
# simply define a silu function
def silu(input):
    for i in range(rowSizeG*colSizeG):
        if input[i][0] < 0:
            input[i][0] = 0
        input[i][0] = input[i][0] + ReLUValue
    return input
    
    
    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def make_fc(num_layers, num_features, num_targets=1,
            activation_fn = nn.ReLU,intermediate_size=512, regularizers = True):
    net_layers = [nn.Linear(num_features, intermediate_size),activation_fn()]
    for hidden in range(num_layers-2):
        net_layers.append(nn.Linear(intermediate_size, intermediate_size))
        net_layers.append(activation_fn())
    net_layers.append(nn.Linear(intermediate_size, num_targets))
    net_layers.append(activation_fn())
    return nn.Sequential(*net_layers)
        

class MyCustomDataset():
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def __len__(self):
        return len(self.value)

    def __getitem__(self, idx):
        return self.feature[idx], self.value[idx]


import sys
import ip_model_whole as ip_model_wholeFile
from ip_model_whole import IPOfunc

class Intopt:
    def __init__(self, c, h, A, b, penalty, n_features, num_layers=5, smoothing=False, thr=0.1, max_iter=None, method=1, mu0=None,
                 damping=0.5, target_size=1, epochs=8, optimizer=optim.Adam,
                 batch_size=rowSizeG*colSizeG, **hyperparams):
        self.c = c
        self.h = h
        self.A = A
        self.b = b
        self.penalty = penalty
        self.target_size = target_size
        self.n_features = n_features
        self.damping = damping
        self.num_layers = num_layers

        self.smoothing = smoothing
        self.thr = thr
        self.max_iter = max_iter
        self.method = method
        self.mu0 = mu0

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.hyperparams = hyperparams
        self.epochs = epochs
        # print("embedding size {} n_features {}".format(embedding_size, n_features))

#        self.model = Net(n_features=n_features, target_size=target_size)
        self.model = make_fc(num_layers=self.num_layers,num_features=n_features)
        #self.model.apply(weight_init)
#        w1 = self.model[0].weight
#        print(w1)

        self.optimizer = optimizer(self.model.parameters(), **hyperparams)

    def fit(self, feature, value):
        logging.info("Intopt")
        train_df = MyCustomDataset(feature, value)

        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='mean')
        grad_list = np.zeros(self.epochs)
        for e in range(self.epochs):
          total_loss = 0
#          for parameters in self.model.parameters():
#            print(parameters)
          if e < 10:
            lr = 1e-1
            #print('stage 1')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            for feature, value in train_dl:
                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
                #print(op)
                
                loss = criterion(op, value)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            #print("Epoch{} ::loss {} ->".format(e,total_loss))
                
          else:
#            if e > 4:
#                for param_group in self.optimizer.param_groups:
#                    param_group['lr'] = 1e-10
            #print('stage 2')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            
            corr_obj_list = []
            num = 0
            batchCnt = 0
            loss = Variable(torch.tensor(0.0, dtype=torch.double), requires_grad=True)
            for feature, value in train_dl:
                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
                while torch.min(op) <= 0 or torch.isnan(op).any() or torch.isinf(op).any():
                    self.optimizer.zero_grad()
#                    self.model.__init__(self.n_features, self.target_size)
                    self.model = make_fc(num_layers=self.num_layers,num_features=self.n_features)
                    op = self.model(feature).squeeze()
  
                price = np.zeros(colSizeG)
                cap = np.zeros(rowSizeG)
                penaltyVector = np.zeros(colSizeG)
                for i in range(colSizeG):
                    price[i] = self.c[i+num*colSizeG]
                    penaltyVector[i] = self.penalty[i+num*colSizeG]
                for j in range(rowSizeG):
                    cap[j] = self.h[j+num*rowSizeG]
                
                c_torch = torch.from_numpy(price).float()
                h_torch = torch.from_numpy(cap).float()
                A_torch = torch.from_numpy(self.A).float()
                b_torch = torch.from_numpy(self.b).float()
                penalty_torch = torch.from_numpy(penaltyVector).float()
                
                cntG = 0
                G_torch = torch.zeros((rowSizeG, colSizeG))
                for i in range(rowSizeG):
                    for j in range(colSizeG):
                        G_torch[i][j] = value[cntG]
                        cntG = cntG + 1
                
                cntOp = 0
                op_torch = torch.zeros((rowSizeG, colSizeG))
                for i in range(rowSizeG):
                    for j in range(colSizeG):
                        op_torch[i][j] = op[cntOp]
                        cntOp = cntOp + 1
                
#                print("c: ", price, "h: ", cap)
#                print(G_torch)
#                print(op_torch)
                x_s2 = IPOfunc(A=A_torch, b=b_torch, h=-h_torch, c=c_torch, GTrue=-G_torch, penalty=penalty_torch, max_iter=self.max_iter, thr=self.thr, damping=self.damping,
                            smoothing=self.smoothing)(-op_torch)
                #print(c_torch.shape, G_torch.shape, x.shape)    # torch.Size([242]) torch.Size([43, 242]) torch.Size([242])
#                newLoss = (x * c_torch).sum()
                x_s1 = ip_model_wholeFile.x_s1
#                print(x_s2 - x_s1, value)
                G_real_numpy = value.numpy()
#                x_true = get_xTrue(self.c, G_real_numpy, self.h, 1)
#                x_true = torch.from_numpy(x_true)
#                newLoss = (x * c_torch).sum() + torch.dot(torch.mul(c_torch, penalty), torch.mul(x, 1-1/ip_model_wholeFile.violateFactor))
                newLoss = torch.dot(penalty_torch * c_torch, abs(x_s2-x_s1).float()) + (x_s2 * c_torch).sum()
#                print(newLoss)
#                print(penaltyVector)
                corr_obj_list.append(newLoss.item())
                loss = loss + newLoss
                batchCnt = batchCnt + 1
#                print(loss)
#                loss = torch.dot(-c_torch, x)
#                print(loss.shape)
                  
#                print(x)
                #loss = -(x * value).mean()
                #loss = Variable(loss, requires_grad=True)
                total_loss += newLoss.item()
                # op.retain_grad()
                #print(loss)
                
                newLoss.backward()
                #print("backward1")
                self.optimizer.step()
                
                # when training size is large
#                if batchCnt % 10 == 0:
#                    newLoss.backward()
#                    #print("backward1")
#                    self.optimizer.step()
                num = num + 1
            total_loss = total_loss/trainSize
          
          logging.info("EPOCH Ends")
          #print("Epoch{}".format(e))
          print("Epoch{} ::loss {} ->".format(e,total_loss))
#          print(corr_obj_list)
          grad_list[e] = total_loss
#          if total_loss < 560:
#            for param_group in self.optimizer.param_groups:
#                param_group['lr'] = 1e-3
#          for param_group in self.optimizer.param_groups:
#            print(param_group['lr'])
          #if e > 1 and grad_list[e] >= grad_list[e-1] and grad_list[e-1] >= grad_list[e-2]:
#          if e > 0 and abs(grad_list[e] - grad_list[e-1]) <= 0.01:
#            break
          if e > 0 and grad_list[e] >= grad_list[e-1]:
            break
#          if total_loss > -200000:
#            break
#          else:
#            currentBestLoss = total_loss
#          if total_loss > -500:
#            break
#           print(self.val_loss(valid_econ, valid_prop))
          # print("______________")

    def val_loss(self, feature, value):
        valueTemp = value.numpy()
        test_instance = len(valueTemp) / self.batch_size
        real_obj = actual_obj(self.c, value, self.h, n_instance=int(test_instance))
#        print(real_obj)
#        print(np.mean(real_obj))

        self.model.eval()
        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(feature, value)
        valid_dl = data_utils.DataLoader(valid_df, batch_size=self.batch_size, shuffle=False)

        obj_list = []
        corr_obj_list = []
        predVal = torch.zeros(len(valueTemp))
        
        num = 0
        for feature, value in valid_dl:
            op = self.model(feature).squeeze()
            for i in range(rowSizeG*colSizeG):
                predVal[i+num*rowSizeG*colSizeG] = op[i]
#            print(op)
            loss = criterion(op, value)

            price = np.zeros(colSizeG)
            cap = np.zeros(rowSizeG)
            penaltyVector = np.zeros(colSizeG)
            for i in range(colSizeG):
                price[i] = self.c[i+num*colSizeG]
                penaltyVector[i] = self.penalty[i+num*colSizeG]
            for j in range(rowSizeG):
                cap[j] = self.h[j+num*rowSizeG]
            
            cntG = 0
            realG = np.zeros((rowSizeG, colSizeG))
            for i in range(rowSizeG):
                for j in range(colSizeG):
                    realG[i][j] = value[cntG]
                    cntG = cntG + 1
            
            cntOp = 0
            predG = np.zeros((rowSizeG, colSizeG))
            for i in range(rowSizeG):
                for j in range(colSizeG):
                    predG[i][j] = op[cntOp]
                    cntOp = cntOp + 1
            
            price = price.tolist()
            cap = cap.tolist()
            
#            print("price: ", price, "cap: ", cap, "realG: ", realG, "predG: ", predG)
            corrrlst = correction_single_obj(price, realG, predG, cap, penaltyVector)
            corr_obj_list.append(corrrlst)
            num = num + 1
            

        self.model.train()
#        print(corr_obj_list)
#        print(np.mean(corr_obj_list))
#        return prediction_loss, abs(np.array(obj_list) - real_obj)
        return abs(np.array(corr_obj_list) - real_obj), predVal


#c_dataTemp = np.loadtxt('KS_c.txt')
#c_data = c_dataTemp[:itemNum]

A_data = np.zeros((2, colSizeG))
b_data = np.zeros(2)

h_train = np.zeros(trainSize*rowSizeG)
h_test = np.zeros(trainSize*rowSizeG)
for i in range(trainSize*rowSizeG):
    h_train[i] = cap[i%rowSizeG]
    h_test[i] = cap[i%rowSizeG]

startmark = 0
endmark = startmark + 1

print("*** HSD ****")

#for testmark in range(startmark, endmark):
    #recordFile = open('record(' + str(testmark) + ').txt', 'a')
testTime = 10
recordBest = np.zeros((1, testTime))

stopCriterion = 0
if penaltyTerm == 0.25:
    stopCriterion = 50
elif penaltyTerm == 0.5:
    stopCriterion = 65
elif penaltyTerm == 1:
    stopCriterion = 90
elif penaltyTerm == 2:
    stopCriterion = 120
elif penaltyTerm == 4:
    stopCriterion = 160
elif penaltyTerm == 8:
    stopCriterion = 180
    

for testi in range(startmark, endmark):
    print(testi)
    
    c_train = np.loadtxt('./data/brass/train_prices/train_prices(' + str(testi) + ').txt')
    x_train = np.loadtxt('./data/brass/train_features/train_features(' + str(testi) + ').txt')
    y_train = np.loadtxt('./data/brass/train_weights/train_weights(' + str(testi) + ').txt')
    penalty_train = np.loadtxt('./data/brass/train_penalty' + str(penaltyTerm) + '/train_penalty(' + str(testi) + ').txt')
    feature_train = torch.from_numpy(x_train).float()
    value_train = torch.from_numpy(y_train).float()
    meanVal = np.mean(y_train)

    c_test = np.loadtxt('./data/brass/test_prices/test_prices(' + str(testi) + ').txt')
    x_test = np.loadtxt('./data/brass/test_features/test_features(' + str(testi) + ').txt')
    y_test = np.loadtxt('./data/brass/test_weights/test_weights(' + str(testi) + ').txt')
    penalty_test = np.loadtxt('./data/brass/test_penalty' + str(penaltyTerm) + '/test_penalty(' + str(testi) + ').txt')
    feature_test = torch.from_numpy(x_test).float()
    value_test = torch.from_numpy(y_test).float()

    start = time.time()
    damping = 1e-2
    thr = 1e-3
#    lr = 5e-7
    lr = 5e-7
    bestTrainCorrReg = float("inf")
    for j in range(3):
        clf = Intopt(c_train, h_train, A_data, b_data, penalty_train, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=40)
        clf.fit(feature_train, value_train)
        train_rslt, predTrainVal = clf.val_loss(feature_train, value_train)
        avgTrainCorrReg = np.mean(train_rslt)
        trainHSD_rslt = ' train: ' + str(np.mean(train_rslt))

        if avgTrainCorrReg < bestTrainCorrReg:
            bestTrainCorrReg = avgTrainCorrReg
            torch.save(clf.model.state_dict(), 'model.pkl')
        print(trainHSD_rslt)
        
        if avgTrainCorrReg < stopCriterion:
            break

    clfBest = Intopt(c_test, h_test, A_data, b_data, penalty_test, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=20)
    clfBest.model.load_state_dict(torch.load('model.pkl'))

    val_rslt, predTestVal = clfBest.val_loss(feature_test, value_test)
    end = time.time()
    HSD_rslt = ' test: ' + str(np.mean(val_rslt))
    print(HSD_rslt)
    print ('Elapsed time: ' + str(end-start))
    recordBest[0][testi] = np.sum(val_rslt)
    
    predTestVal = predTestVal.detach().numpy()
    predValue = np.zeros((predTestVal.size, 2))
    for i in range(predTestVal.size):
        predValue[i][0] = value_test[i]
        predValue[i][1] = predTestVal[i]
        
    np.savetxt('./data/brass/2S_weights/2S_weights' + str(penaltyTerm) + '(' + str(testi) + ').txt', predValue, fmt="%.2f")
    

print(recordBest)
