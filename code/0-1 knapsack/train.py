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
from sklearn.metrics import mean_squared_error
import gurobipy as gp
import logging
import copy
from collections import defaultdict
import joblib
import gurobipy as gp
from gurobipy import GRB

capacity = 100
purchase_fee = 0.2
compensation_fee = 0.21

itemNum = 10
featureNum = 4096
trainSize = 700
targetNum = 2

def get_xTrue(valueTemp, cap, weightTemp, n_instance):
    obj_list = []
    selectedNum_list = []
    for num in range(n_instance):
        weight = np.zeros(itemNum)
        value = np.zeros(itemNum)
        cnt = num * itemNum
        for i in range(itemNum):
            weight[i] = weightTemp[cnt]
            value[i] = valueTemp[cnt]
            cnt = cnt + 1
        weight = weight.tolist()
        value = value.tolist()
        
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.BINARY, name='x')
        m.setObjective(purchase_fee * x.prod(value), GRB.MAXIMIZE)
        m.addConstr((x.prod(weight)) <= cap)
#        for i in range(itemNum):
#            m.addConstr((x.prod(weight[i])) <= cap)

        m.optimize()
        sol = np.zeros(itemNum)
        for i in range(itemNum):
            sol[i] = x[i].x
            
        objective = m.objVal
#        print("TOV: ", sol, objective)
        
    return sol

def get_Xs1Xs2(realPrice, predPrice, cap, realWeightTemp, predWeightTemp):
#    print("realPrice: ", realPrice)
    realWeight = np.zeros(itemNum)
    predWeight = np.zeros(itemNum)
    realPriceNumpy = np.zeros(itemNum)
    for i in range(itemNum):
        realWeight[i] = realWeightTemp[i]
        predWeight[i] = predWeightTemp[i]
        realPriceNumpy[i] = realPrice[i]
        
    if min(predWeight) >= 0:
        predWeight = predWeight.tolist()
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.BINARY, name='x')
        m.setObjective(purchase_fee * x.prod(predPrice), GRB.MAXIMIZE)
        m.addConstr((x.prod(predWeight)) <= cap)

        m.optimize()
        predSol = np.zeros(itemNum,dtype='i')
        for i in range(itemNum):
            predSol[i] = x[i].x
            
        objective1 = m.objVal
#        print("Stage 1: ", predSol, objective1)

        # Stage 2:
        realWeight = realWeight.tolist()
        m2 = gp.Model()
        m2.setParam('OutputFlag', 0)
        x = m2.addVars(itemNum, vtype=GRB.BINARY, name='x')
        sigma = m2.addVars(itemNum, vtype=GRB.BINARY, name='sigma')

        OBJ = purchase_fee * x.prod(realPrice)
        for i in range(itemNum):
            OBJ = OBJ - compensation_fee * realPrice[i] * sigma[i]
        m2.setObjective(OBJ, GRB.MAXIMIZE)

        m2.addConstr((x.prod(realWeight) - sigma.prod(realWeight)) <= cap)
        for i in range(itemNum):
            m2.addConstr(x[i] == predSol[i])
            m2.addConstr(x[i] >= sigma[i])
        try:
            m2.optimize()
            objective = m2.objVal
            sol = np.zeros(itemNum)
            for i in range(itemNum):
                sol[i] = x[i].x - sigma[i].x
        except:
            print(predPrice, predWeight, realPrice, realWeight, predSol)
#        print("Stage 2: ", sol, objective)

    return predSol,sol
    

def actual_obj(valueTemp, cap, weightTemp, n_instance):
    obj_list = []
    selectedNum_list = []
    for num in range(n_instance):
        weight = np.zeros(itemNum)
        value = np.zeros(itemNum)
        cnt = num * itemNum
        for i in range(itemNum):
            weight[i] = weightTemp[cnt]
            value[i] = valueTemp[cnt]
            cnt = cnt + 1
        weight = weight.tolist()
        value = value.tolist()
        
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.BINARY, name='x')
        m.setObjective(purchase_fee * x.prod(value), GRB.MAXIMIZE)
        m.addConstr((x.prod(weight)) <= cap)
#        for i in range(itemNum):
#            m.addConstr((x.prod(weight[i])) <= cap)

        m.optimize()
        sol = []
        selectedItemNum = 0
        for i in range(itemNum):
            sol.append(x[i].x)
            if x[i].x == 1:
              selectedItemNum = selectedItemNum + 1
        objective = m.objVal
        obj_list.append(objective)
        selectedNum_list.append(selectedItemNum)
        # print(selectedItemNum)
#        print("TOV: ", sol, objective)
        
    return np.array(obj_list)


def correction_single_obj(realPrice, predPrice, cap, realWeightTemp, predWeightTemp):
#    print("realPrice: ", realPrice, "predPrice: ", predPrice)
    realWeight = np.zeros(itemNum)
    predWeight = np.zeros(itemNum)
    realPriceNumpy = np.zeros(itemNum)
    for i in range(itemNum):
        realWeight[i] = realWeightTemp[i]
        predWeight[i] = predWeightTemp[i]
        realPriceNumpy[i] = realPrice[i]
        
    if min(predWeight) >= 0:
        predWeight = predWeight.tolist()
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.BINARY, name='x')
        m.setObjective(purchase_fee * x.prod(predPrice), GRB.MAXIMIZE)
        m.addConstr((x.prod(predWeight)) <= cap)

        m.optimize()
        predSol = np.zeros(itemNum,dtype='i')
        x1_selectedItemNum = 0
        for i in range(itemNum):
            predSol[i] = x[i].x
            if x[i].x == 1:
              x1_selectedItemNum = x1_selectedItemNum + 1
        objective1 = m.objVal
#        print("Stage 1: ", predSol, objective1)

        # Stage 2:
        realWeight = realWeight.tolist()
        m2 = gp.Model()
        m2.setParam('OutputFlag', 0)
        x = m2.addVars(itemNum, vtype=GRB.BINARY, name='x')
        sigma = m2.addVars(itemNum, vtype=GRB.BINARY, name='sigma')

        OBJ = purchase_fee * x.prod(realPrice)
        for i in range(itemNum):
            OBJ = OBJ - compensation_fee * realPrice[i] * sigma[i]
        m2.setObjective(OBJ, GRB.MAXIMIZE)

        m2.addConstr((x.prod(realWeight) - sigma.prod(realWeight)) <= cap)
        for i in range(itemNum):
            m2.addConstr(x[i] == predSol[i])
            m2.addConstr(x[i] >= sigma[i])
        
        try:
            m2.optimize()
            objective = m2.objVal
            sol = []
            x2_selectedItemNum = 0
            for i in range(itemNum):
                sol.append(x[i].x - sigma[i].x)
                if x[i].x - sigma[i].x == 1:
                  x2_selectedItemNum = x2_selectedItemNum + 1
    #        print("Stage 2: ", sol, objective)
        except:
            print(predPrice, predWeight, realPrice, realWeight, predSol)

    return objective

    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def make_fc(num_layers, num_features, num_targets=targetNum,
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
    def __init__(self, c, h, A, b, purchase_fee, compensation_fee, n_features, num_layers=5, smoothing=False, thr=0.1, max_iter=None, method=1, mu0=None,
                 damping=0.5, target_size=targetNum, epochs=8, optimizer=optim.Adam,
                 batch_size=itemNum, **hyperparams):
        self.c = c
        self.h = h
        self.A = A
        self.b = b
        self.target_size = target_size
        self.n_features = n_features
        self.damping = damping
        self.num_layers = num_layers
        self.purchase_fee = purchase_fee
        self.compensation_fee = compensation_fee

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
          if e < 5:
            #print('stage 1')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            for feature, value in train_dl:
                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
#                print(feature, value, op)
#                print(feature.shape, value.shape, op.shape)
                # targetNum=1: torch.Size([10, 4096]) torch.Size([10]) torch.Size([10])
                # targetNum=2: torch.Size([10, 4096]) torch.Size([10, 2]) torch.Size([10, 2])
#                print(value, op)
                
                loss = criterion(op, value)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            grad_list[e] = total_loss
            print("Epoch{} ::loss {} ->".format(e,total_loss))
                
          else:
            #print('stage 2')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            
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
  
                price = np.zeros(itemNum)
                for i in range(itemNum):
                    price[i] = self.c[i+num*itemNum]
                    op[i] = op[i]
                    
                c_torch = torch.from_numpy(price).float()
                h_torch = torch.from_numpy(self.h).float()
                A_torch = torch.from_numpy(self.A).float()
                b_torch = torch.from_numpy(self.b).float()
                
                G_torch = torch.zeros((itemNum+1, itemNum))
                for i in range(itemNum):
                    G_torch[i][i] = 1
                G_torch[itemNum] = value[:, 1]
                trueWeight = value[:, 1]
                
#                op_torch = torch.zeros((itemNum+1, itemNum))
#                for i in range(itemNum):
#                    op_torch[i][i] = 1
#                op_torch[itemNum] = op
                
#                print(G_torch)
#                print(op_torch)
                x_s2 = IPOfunc(A=A_torch, b=b_torch, h=h_torch, cTrue=-c_torch, GTrue=G_torch, purchase_fee=self.purchase_fee, compensation_fee=self.compensation_fee, max_iter=self.max_iter, thr=self.thr, damping=self.damping,
                            smoothing=self.smoothing)(op)
                x_s1 = ip_model_wholeFile.x_s1
                
#                trueWeight = trueWeight.numpy()
##                print(x, c_torch)
##                newLoss = (x * c_torch).sum() + torch.dot(torch.mul(c_torch, penalty), torch.mul(x, 1-1/ip_model_wholeFile.violateFactor))
#                x_true = get_xTrue(price, capacity, trueWeight, 1)
#                x_true = torch.from_numpy(x_true)
#                price = price.tolist()
#                predPrice = op[:, 0].detach().tolist()
#                predWeight = op[:, 1].detach().tolist()
#                x_s1_true, x_s2_true = get_Xs1Xs2(price, predPrice, capacity, trueWeight, predWeight)
##                print(x_s1_true,x_s2_true)
#                x_s1_true = torch.from_numpy(x_s1_true)
#                x_s2_true = torch.from_numpy(x_s2_true)
                
                newLoss = - (purchase_fee * (x_s2 * c_torch).sum() - (compensation_fee - purchase_fee) * torch.dot(c_torch, abs(x_s2-x_s1).float()))
#                print(x_s2,x_s1,newLoss)
#                newLoss.data = purchase_fee * (x_true * c_torch).sum() - (purchase_fee * (x_s2_true * c_torch).sum() - (compensation_fee - purchase_fee) * torch.dot(c_torch, abs(x_s2_true-x_s1_true).float()))
#                print(x_s2_true,x_s1_true,newLoss)
#                print(newLoss)
#                newLoss = - (x * c_torch).sum()
#                loss = loss - (x * c_torch).sum()
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
                if batchCnt % 70 == 0:
                    print(newLoss)
#                    newLoss.backward()
#                    #print("backward1")
#                    self.optimizer.step()
                num = num + 1
            grad_list[e] = total_loss/trainSize
            print("Epoch{} ::loss {} ->".format(e,grad_list[e]))
                
          
          logging.info("EPOCH Ends")
          #print("Epoch{}".format(e))
#          for param_group in self.optimizer.param_groups:
#            print(param_group['lr'])
          if e > 1 and abs(grad_list[e] - grad_list[e-1]) <= 0.01:
            break
            

    def val_loss(self, cap, feature, value):
        valueTemp = value.numpy()
#        test_instance = len(valueTemp) / self.batch_size
        test_instance = np.size(valueTemp, 0) / self.batch_size
#        itemVal = self.c.tolist()
        itemVal = self.c
        real_obj = actual_obj(itemVal, cap, value[:, 1], n_instance=int(test_instance))
#        print(np.sum(real_obj))

        self.model.eval()
        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(feature, value)
        valid_dl = data_utils.DataLoader(valid_df, batch_size=self.batch_size, shuffle=False)

        obj_list = []
        corr_obj_list = []
        len = np.size(valueTemp, 0)
        predVal = torch.zeros((len, 2))
        
        num = 0
        for feature, value in valid_dl:
            op = self.model(feature).squeeze()
#            print(op)
            loss = criterion(op, value)

            realWT = {}
            predWT = {}
            realPrice = {}
            predPrice = {}
            for i in range(itemNum):
                realWT[i] = value[i][1]
                predWT[i] = op[i][1]
                realPrice[i] = value[i][0]
                predPrice[i] = op[i][0]
                predVal[i+num*itemNum][0] = op[i][0]
                predVal[i+num*itemNum][1] = op[i][1]

            corrrlst = correction_single_obj(realPrice, predPrice, cap, realWT, predWT)
            corr_obj_list.append(corrrlst)
            num = num + 1
            

        self.model.train()
#        print(corr_obj_list)
#        print(corr_obj_list-real_obj)
#        print(np.sum(corr_obj_list))
#        return prediction_loss, abs(np.array(obj_list) - real_obj)
        return abs(np.array(corr_obj_list) - real_obj), predVal


#c_dataTemp = np.loadtxt('KS_c.txt')
#c_data = c_dataTemp[:itemNum]

h_data = np.ones(itemNum+1)
h_data[itemNum] = capacity
A_data = np.zeros((2, itemNum))
b_data = np.zeros(2)


print("*** HSD ****")

testTime = 1
recordBest = np.zeros((1, testTime))
stopCriterior = 15

for testi in range(testTime):
    print(testi)
    
    c_train = np.loadtxt('./data/train_prices/train_prices(' + str(testi) + ').txt')
    x_train = np.loadtxt('./data/train_features/train_features(' + str(testi) + ').txt')
    y_train1 = np.loadtxt('./data/train_prices/train_prices(' + str(testi) + ').txt')
    y_train2 = np.loadtxt('./data/train_weights/train_weights(' + str(testi) + ').txt')
    meanPriceValue = np.mean(y_train1)
    meanWeightValue = np.mean(y_train2)

    y_train = np.zeros((y_train1.size, 2))
    for i in range(y_train1.size):
        y_train[i][0] = y_train1[i]
        y_train[i][1] = y_train2[i]
    feature_train = torch.from_numpy(x_train).float()
    value_train = torch.from_numpy(y_train).float()
    
    c_test = np.loadtxt('./data/test_prices/test_prices(' + str(testi) + ').txt')
    x_test = np.loadtxt('./data/test_features/test_features(' + str(testi) + ').txt')
    y_test1 = np.loadtxt('./data/test_prices/test_prices(' + str(testi) + ').txt')
    y_test2 = np.loadtxt('./data/test_weights/test_weights(' + str(testi) + ').txt')

    y_test = np.zeros((y_test1.size, 2))
    for i in range(y_test1.size):
        y_test[i][0] = y_test1[i]
        y_test[i][1] = y_test2[i]
    feature_test = torch.from_numpy(x_test).float()
    value_test = torch.from_numpy(y_test).float()
    
    start = time.time()
    damping = 1e-2
    thr = 1e-3
    lr = 1e-5
    bestTrainCorrReg = float("inf")
    for j in range(3):
        clf = Intopt(c_train, h_data, A_data, b_data, purchase_fee, compensation_fee, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=20)
        clf.fit(feature_train, value_train)
        train_rslt, predTrainVal = clf.val_loss(capacity, feature_train, value_train)
        avgTrainCorrReg = np.mean(train_rslt)
        trainHSD_rslt = 'train: ' + str(np.mean(train_rslt))

        if avgTrainCorrReg < bestTrainCorrReg:
            bestTrainCorrReg = avgTrainCorrReg
            torch.save(clf.model.state_dict(), 'model.pkl')
        print(trainHSD_rslt)
        
        if avgTrainCorrReg < stopCriterior:
            break


    clfBest = Intopt(c_test, h_data, A_data, b_data, purchase_fee, compensation_fee, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=8)
    clfBest.model.load_state_dict(torch.load('model.pkl'))

    val_rslt, predTestVal = clfBest.val_loss(capacity, feature_test, value_test)
    end = time.time()

    predTestVal = predTestVal.detach().numpy()
#    print(predTestVal.shape)
    predTestVal1 = predTestVal[:, 0]
    predTestVal2 = predTestVal[:, 1]
    predValuePrice = np.zeros((predTestVal1.size, 2))
    for i in range(predTestVal1.size):
#        predValue[i][0] = int(i/itemNum)
        predValuePrice[i][0] = y_test1[i]
        predValuePrice[i][1] = predTestVal1[i]
    np.savetxt('./data/2S_prices/2S_prices_cap' + str(capacity) + '_compensation' + str(compensation_fee) + '(' + str(testi) + ').txt', predValuePrice, fmt="%.2f")
    predValueWeight = np.zeros((predTestVal2.size, 2))
    for i in range(predTestVal2.size):
#        predValue[i][0] = int(i/itemNum)
        predValueWeight[i][0] = y_test2[i]
        predValueWeight[i][1] = predTestVal2[i]
    np.savetxt('./data/2S_weights/2S_weights_cap' + str(capacity) + '_compensation' + str(compensation_fee) + '(' + str(testi) + ').txt', predValueWeight, fmt="%.2f")
    
    HSD_rslt = 'test: ' + str(np.mean(val_rslt))
    print(HSD_rslt)
    print ('Elapsed time: ' + str(end-start))
    recordBest[0][testi] = np.sum(val_rslt)

print(recordBest)
