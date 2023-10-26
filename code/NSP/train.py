# use ip_model_whole(logKKT(Gh)_hUn).py

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

nurse_num = 15
day_num = 7
shift_num = 3
decision_num = nurse_num * day_num * shift_num
day_shift_num = day_num * shift_num
penaltyTerm = 0.25
featureNum = 8
train_loss = 0

def gen_matrix(nurse_num,day_num,shift_num,serve_patient_num,decision_num,day_shift_num):
    # Each nurse must be scheduled for exactly one shift per day
    A = np.zeros((nurse_num*day_num, decision_num))
    for i in range(nurse_num):
        for j in range(day_num):
            for q in range(shift_num):
                A[i*day_num+j][i*day_shift_num+shift_num*j+q] = 1
    #print(A)
    b = np.ones(nurse_num*day_num)

    # Each schedule must satisfy the patients' need
    G1 = np.zeros((day_shift_num, decision_num))
    for j in range(day_shift_num):
        for i in range(nurse_num):
            G1[j][i*day_shift_num+j] = -serve_patient_num[i]
#    G1 = serve_patient_num * G1
    #print(G1)
    # No nurse may be scheduled to work a night shift followed immendiately by a morning shift
    G2 = np.zeros((nurse_num*(day_num-1), decision_num))
    for i in range(nurse_num):
        for j in range(day_num-1):
            G2[i*(day_num-1)+j][i*day_shift_num+shift_num*(j+1)-1] = 1
            G2[i*(day_num-1)+j][i*day_shift_num+shift_num*(j+1)] = 1
    #print(G2)
    h2 = np.ones(nurse_num*(day_num-1))
    G3 = np.identity(decision_num)
    h3 = np.ones(decision_num)
    
    G = np.concatenate([G1, G2, G3], axis=0)
    
    return A,b,G,h2,h3

def get_xTrue(c, A, b, G, real_patient_num, h2, h3, n_instance):
    obj_list = []
    rowSizeA = A.shape[0]
    rowSizeG = G.shape[0]
    c = c.tolist()
    A = A.tolist()
    b = b.tolist()
    G = G.tolist()
    
    for num in range(n_instance):
        h1 = np.zeros(day_shift_num)
        cnt = num * day_shift_num
        for i in range(day_shift_num):
            h1[i] = -real_patient_num[cnt]
            cnt = cnt + 1
        h = np.concatenate([h1, h2, h3], axis=0)
#        print(h1)
        h = h.tolist()

        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(decision_num, vtype=GRB.BINARY, name='x')
        
        m.setObjective(x.prod(c), GRB.MAXIMIZE)
        for i in range(rowSizeA):
            m.addConstr(x.prod(A[i]) == b[i])
        for j in range(rowSizeG):
            m.addConstr(x.prod(G[j]) <= h[j])

        m.optimize()
        sol = np.zeros(decision_num)
        try:
            for i in range(decision_num):
                sol[i] = x[i].x
        except:
            for i in range(decision_num):
                sol[i] = 0
        
    return sol
    
def get_Xs1Xs2(c, A, b, G, real_patient_num, pre_patient_num, h2, h3, penalty):
    rowSizeA = A.shape[0]
    rowSizeG = G.shape[0]
    c = c.tolist()
    A = A.tolist()
    b = b.tolist()
    G = G.tolist()
    
    pre_h1 = np.zeros(day_shift_num)
    real_h1 = np.zeros(day_shift_num)
    for i in range(day_shift_num):
        pre_h1[i] = -pre_patient_num[i]
        real_h1[i] = -real_patient_num[i]
    pre_h = np.concatenate([pre_h1, h2, h3], axis=0)
    real_h = np.concatenate([real_h1, h2, h3], axis=0)
    pre_h = pre_h.tolist()
    real_h1 = real_h1.tolist()

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    x = m.addVars(decision_num, vtype=GRB.BINARY, name='x')
    
    m.setObjective(x.prod(c), GRB.MAXIMIZE)
    for i in range(rowSizeA):
        m.addConstr(x.prod(A[i]) == b[i])
    for j in range(rowSizeG):
        m.addConstr(x.prod(G[j]) <= pre_h[j])

    m.optimize()
    predSol = np.zeros(decision_num)
    try:
        for i in range(decision_num):
            predSol[i] = x[i].x
    except:
        for i in range(decision_num):
            predSol[i] = 0

    # Stage 2:
    m2 = gp.Model()
    m2.setParam('OutputFlag', 0)
    x = m2.addVars(decision_num, vtype=GRB.BINARY, name='x')
    sigma = m2.addVars(decision_num, vtype=GRB.BINARY, name='sigma')

    OBJ = x.prod(c)
    for i in range(decision_num):
        OBJ = OBJ - penalty[i] * (5 - c[i]) * (5 - c[i]) * sigma[i]
    m2.setObjective(OBJ, GRB.MAXIMIZE)

    for i in range(rowSizeA):
        m2.addConstr(x.prod(A[i]) == b[i])
    for j in range(rowSizeG):
        m2.addConstr(x.prod(G[j]) <= real_h[j])
    for i in range(decision_num):
        m2.addConstr(sigma[i] >= x[i] - predSol[i])
    

    m2.optimize()
    sol = np.zeros(decision_num)
    try:
        for i in range(decision_num):
            sol[i] = x[i].x
    except:
        for i in range(decision_num):
            sol[i] = 0
        
    return predSol, sol


def actual_obj(c, A, b, G, real_patient_num, h2, h3, n_instance):
    obj_list = []
    rowSizeA = A.shape[0]
    rowSizeG = G.shape[0]
    c = c.tolist()
    A = A.tolist()
    b = b.tolist()
    G = G.tolist()
    
    for num in range(n_instance):
        h1 = np.zeros(day_shift_num)
        cnt = num * day_shift_num
        for i in range(day_shift_num):
            h1[i] = -real_patient_num[cnt]
            cnt = cnt + 1
        h = np.concatenate([h1, h2, h3], axis=0)
#        print(h1)
        h = h.tolist()

        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(decision_num, vtype=GRB.BINARY, name='x')
        
        m.setObjective(x.prod(c), GRB.MAXIMIZE)
        for i in range(rowSizeA):
            m.addConstr(x.prod(A[i]) == b[i])
        for j in range(rowSizeG):
            m.addConstr(x.prod(G[j]) <= h[j])

        m.optimize()
        try:
            sol = []
            for i in range(decision_num):
                sol.append(x[i].x)
            objective = m.objVal
        except:
            objective = 0

        obj_list.append(objective)
        
    return np.array(obj_list)
    
    
def correction_single_obj(c, A, b, G, real_patient_num, pre_patient_num, h2, h3, penalty):
    rowSizeA = A.shape[0]
    rowSizeG = G.shape[0]
    c = c.tolist()
    A = A.tolist()
    b = b.tolist()
    G = G.tolist()
    
    pre_h1 = np.zeros(day_shift_num)
    real_h1 = np.zeros(day_shift_num)
    for i in range(day_shift_num):
        pre_h1[i] = -pre_patient_num[i]
        real_h1[i] = -real_patient_num[i]
    pre_h = np.concatenate([pre_h1, h2, h3], axis=0)
    real_h = np.concatenate([real_h1, h2, h3], axis=0)
    pre_h = pre_h.tolist()
    real_h1 = real_h1.tolist()

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    x = m.addVars(decision_num, vtype=GRB.BINARY, name='x')
    
    m.setObjective(x.prod(c), GRB.MAXIMIZE)
    for i in range(rowSizeA):
        m.addConstr(x.prod(A[i]) == b[i])
    for j in range(rowSizeG):
        m.addConstr(x.prod(G[j]) <= pre_h[j])

    m.optimize()
    
    try:
        predSol = []
        for i in range(decision_num):
            predSol.append(x[i].x)
    except:
        predSol = []
        for i in range(decision_num):
            predSol.append(0)

    # Stage 2:
    m2 = gp.Model()
    m2.setParam('OutputFlag', 0)
    x = m2.addVars(decision_num, vtype=GRB.BINARY, name='x')
    sigma = m2.addVars(decision_num, vtype=GRB.BINARY, name='sigma')

    OBJ = x.prod(c)
    for i in range(decision_num):
        OBJ = OBJ - penalty[i] * (5 - c[i]) * (5 - c[i]) * sigma[i]
    m2.setObjective(OBJ, GRB.MAXIMIZE)

    for i in range(rowSizeA):
        m2.addConstr(x.prod(A[i]) == b[i])
    for j in range(rowSizeG):
        m2.addConstr(x.prod(G[j]) <= real_h[j])
    for i in range(decision_num):
        m2.addConstr(sigma[i] >= x[i] - predSol[i])
    

    m2.optimize()
    sol = []
    try:
        for i in range(decision_num):
            sol.append(x[i].x)
        objective = m2.objVal
    except:
        for i in range(decision_num):
            sol.append(0)
        objective = 0
    
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

def make_fc(num_layers, num_features, num_targets=1,
            activation_fn = nn.ReLU,intermediate_size=2*featureNum, regularizers = True):
    net_layers = [nn.Linear(num_features, intermediate_size),
         activation_fn()]
    for hidden in range(num_layers-2):
        net_layers.append(nn.Linear(intermediate_size, intermediate_size))
        net_layers.append(activation_fn())
    net_layers.append(nn.Linear(intermediate_size, num_targets))
    net_layers.append(nn.ReLU())
    return nn.Sequential(*net_layers)

## simply define a silu function
#def silu(input):
#    for i in range(day_shift_num):
#        if input[i] < 0:
#            input[i] = 0
#        input[i] = input[i] + ReLUValue
#    return input
#
## create a class wrapper from PyTorch nn.Module, so
## the function now can be easily used in models
#class SiLU(nn.Module):
#    def __init__(self):
#        super().__init__() # init the base class
#
#    def forward(self, input):
#        return silu(input) # simply apply already implemented SiLU
#
## initialize activation function
#activation_function = SiLU()
#
#def weight_init(m):
#    if isinstance(m, nn.Linear):
#        nn.init.xavier_normal_(m.weight)
#        nn.init.constant_(m.bias, 0)
#
#    elif isinstance(m, nn.Conv2d):
#        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#    elif isinstance(m, nn.BatchNorm2d):
#        nn.init.constant_(m.weight, 1)
#        nn.init.constant_(m.bias, 0)
#
#def make_fc(num_layers, num_features, num_targets=1,
#            activation_fn = nn.ReLU,intermediate_size=2*featureNum, regularizers = True):
#    net_layers = [nn.Linear(num_features, intermediate_size),activation_fn()]
#    for hidden in range(num_layers-2):
#        net_layers.append(nn.Linear(intermediate_size, intermediate_size))
#        net_layers.append(activation_fn())
#    net_layers.append(nn.Linear(intermediate_size, num_targets))
#    net_layers.append(activation_function)
#    return nn.Sequential(*net_layers)
        

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
from ip_model_whole import solve_LP

class Intopt:
    def __init__(self, c, G, A, b, h2, h3, penalty, n_features, num_layers=4, smoothing=False, thr=0.1, max_iter=None, method=1, mu0=None,
                 damping=1e-7, target_size=1, epochs=8, optimizer=optim.Adam,
                 batch_size=day_shift_num, **hyperparams):
        self.c = c
        self.G = G
        self.A = A
        self.b = b
        self.h2 = h2
        self.h3 = h3
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
          if e < 0:
            lr = 1e-5
            self.batch_size=2*day_shift_num
            #print('stage 1')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            for feature, value in train_dl:
                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
                while torch.min(op) <= 0 or torch.isnan(op).any() or torch.isinf(op).any():
                    self.optimizer.zero_grad()
#                    self.model.__init__(self.n_features, self.target_size)
                    self.model = make_fc(num_layers=self.num_layers,num_features=self.n_features)
                    op = self.model(feature).squeeze()
                #print(op)
                
                loss = criterion(op, value)
                total_loss += loss.item()
                grad_list[e] = total_loss
                loss.backward()
                self.optimizer.step()
            print("Epoch{} ::loss {} ->".format(e,total_loss))
                
          else:
            self.batch_size=day_shift_num
            #print('stage 2')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            instance_num = 0
            batchCnt = 0
#            train_loss = np.zeros(1)
            
            for feature, value in train_dl:
                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
                while torch.min(op) <= 0 or torch.isnan(op).any() or torch.isinf(op).any():
                    self.optimizer.zero_grad()
#                    self.model.__init__(self.n_features, self.target_size)
                    self.model = make_fc(num_layers=self.num_layers,num_features=self.n_features)
                    op = self.model(feature).squeeze()
    
                penaltyVector = np.zeros(decision_num)
                for i in range(decision_num):
                    penaltyVector[i] = self.penalty[i+instance_num*decision_num]
                
                c_torch = torch.from_numpy(self.c).float()
                G_torch = torch.from_numpy(self.G).float()
                A_torch = torch.from_numpy(self.A).float()
                b_torch = torch.from_numpy(self.b).float()
                h2_torch = torch.from_numpy(self.h2).float()
                h3_torch = torch.from_numpy(self.h3).float()
                penalty_torch = torch.from_numpy(penaltyVector).float()
#                print(h2_torch.shape,h3_torch.shape,op.shape)
  
                x_s2 = IPOfunc(A=A_torch, b=b_torch, G=G_torch, h2=h2_torch, h3=h3_torch, c=-c_torch, h1True=-value, penalty=penalty_torch, max_iter=self.max_iter, thr=self.thr, damping=self.damping,
                            smoothing=self.smoothing)(-op)
                x_s1 = ip_model_wholeFile.x_s1
#                x_true = solve_LP(A_torch,b_torch,G_torch,value,-c_torch)
                h1_real_numpy = value.numpy()
                h1_pre_numpy = op.detach().numpy()
                x_true = get_xTrue(self.c, self.A, self.b, self.G, h1_real_numpy, self.h2, self.h3, 1)
                x_true = torch.from_numpy(x_true)
                x_s1_true, x_s2_true = get_Xs1Xs2(self.c, self.A, self.b, self.G, h1_real_numpy, h1_pre_numpy, self.h2, self.h3, penaltyVector)
                x_s1_true = torch.from_numpy(x_s1_true).float()
                x_s2_true = torch.from_numpy(x_s2_true).float()
                
                penalty_num = torch.zeros(decision_num)
                for i in range(decision_num):
                    if x_s2_true[i] > x_s1_true[i]:
                        penalty_num[i] = penalty_torch[i] * (5 - c_torch[i]) * (5 - c_torch[i])
                loss = (x_true * c_torch).sum() - (x_s2 * c_torch).sum() + penalty_num.sum()
                loss.data = (x_true * c_torch).sum() - (x_s2_true * c_torch).sum() + penalty_num.sum()
                
                if batchCnt % 30 == 0:
                    print("  op: ", op.mean())
                    print("  loss: ", loss)
#                train_loss[instance_num] = loss.detach().numpy()
                batchCnt += 1
                total_loss += loss.item()

                instance_num = instance_num + 1
                if op.mean() > 100 or loss.item() > 200:
                    break
                
            total_loss = total_loss / instance_num
                
            logging.info("EPOCH Ends")
            #print("Epoch{}".format(e))
            #          print(train_loss)
            print("Epoch{} ::loss {} ->".format(e,total_loss))
            grad_list[e] = total_loss
            global train_loss
            train_loss = total_loss
            if grad_list[e] > stopCriterion:
                break
            if e > 0 and grad_list[e] >= grad_list[e-1]:
                break
          # print(self.val_loss(valid_econ, valid_prop))
          # print("______________")

    def val_loss(self, feature, value):
        valueTemp = value.numpy()
#        c_list = self.c.tolist()
#        G_list = self.G.tolist()
        test_instance = len(valueTemp) / self.batch_size
#        test_instance = 1
        real_obj = actual_obj(self.c, self.A, self.b, self.G, value, self.h2, self.h3, n_instance=int(test_instance))
#        print(real_obj)

        self.model.eval()
        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(feature, value)
        valid_dl = data_utils.DataLoader(valid_df, batch_size=self.batch_size, shuffle=False)
        prediction_loss = 0
        corr_obj_list = []
        num = 0

        preVal = np.zeros(90*day_shift_num)
        for feature, value in valid_dl:
            op = self.model(feature).squeeze()
#            print(op)
            loss = criterion(op, value)
            prediction_loss += loss.item()
            
            for i in range(day_shift_num):
                preVal[i+num*day_shift_num] = op[i]

            real_patient = {}
            pre_patient = {}
            for i in range(day_shift_num):
                real_patient[i] = value[i]
                pre_patient[i] = op[i]
            
            penaltyVector = np.zeros(decision_num)
            for i in range(decision_num):
                penaltyVector[i] = self.penalty[i+num*decision_num]
            
            corrrlst = correction_single_obj(self.c, self.A, self.b, self.G, real_patient, pre_patient, self.h2, self.h3, penaltyVector)
            corr_obj_list.append(corrrlst)
            
            num = num + 1

        self.model.train()
#        print("corr_obj_list: ", corr_obj_list)
#        print("2SReg: ", real_obj - np.array(corr_obj_list))
#        return prediction_loss, abs(np.array(obj_list) - real_obj)
        return abs(real_obj - np.array(corr_obj_list)), preVal


ReLUValue = 0
stopCriterion = 0
if penaltyTerm == 0.25:
    stopCriterion = 10
elif penaltyTerm == 0.5:
    stopCriterion = 5
elif penaltyTerm == 4:
    stopCriterion = 25
elif penaltyTerm == 8:
    stopCriterion = 50

print("*** HSD ****")

testTime = 1
recordBest = np.zeros((1, testTime))

for testi in range(testTime):
    print(testi)
    perference = np.loadtxt('./data/preference/preference(' + str(testi) + ').txt')
    c = np.zeros(decision_num)
    for i in range(nurse_num):
        for j in range(day_shift_num):
            c[i*day_shift_num+j] = perference[i][j]
            
    serve_patient_num = np.loadtxt('./data/serve_patient_num/serve_patient_num(' + str(testi) + ').txt')
    A,b,G,h2,h3 = gen_matrix(nurse_num,day_num,shift_num,serve_patient_num,decision_num,day_shift_num)
    
    trainData = np.loadtxt('./data/train/train(' + str(testi) + ').txt')
    penalty_train = np.loadtxt('./data/penalty' + str(penaltyTerm) + '/train_penalty' + str(penaltyTerm) + '/train_penalty(' + str(testi) + ').txt')
#    trainData = np.loadtxt('train.txt')
    x_train = trainData[:, 1:featureNum+1]
    y_train = trainData[:, featureNum+1]
    feature_train = torch.from_numpy(x_train).float()
    value_train = torch.from_numpy(y_train).float()

    testData = np.loadtxt('./data/test/test(' + str(testi) + ').txt')
    penalty_test = np.loadtxt('./data/penalty' + str(penaltyTerm) + '/test_penalty' + str(penaltyTerm) + '/test_penalty(' + str(testi) + ').txt')
#    testData = np.loadtxt('test.txt')
    x_test = testData[:, 1:featureNum+1]
    y_test = testData[:, featureNum+1]
    feature_test = torch.from_numpy(x_test).float()
    value_test = torch.from_numpy(y_test).float()

    damping = 1e-7
    thr = 1e-3
    lr = 1e-1
    #lr = 1e-2
    bestTrainCorrReg = float("inf")
    for j in range(5):
        clf = Intopt(c, G, A, b, h2, h3, penalty_train, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=8)
        clf.fit(feature_train, value_train)
#        train_rslt = clf.val_loss(feature_train, value_train)
#        avgTrainCorrReg = np.mean(train_rslt)
        avgTrainCorrReg = train_loss
    #    trainHSD_rslt = str(testmark) + ' train: ' + str(np.sum(train_rslt[1])) + ' ' + str(np.mean(train_rslt[1]))
#        trainHSD_rslt = ' train: ' + str(np.mean(train_rslt))
        trainHSD_rslt = ' train: ' + str(avgTrainCorrReg)
        print(trainHSD_rslt)
    
        if avgTrainCorrReg < bestTrainCorrReg:
            bestTrainCorrReg = avgTrainCorrReg
            torch.save(clf.model.state_dict(), 'model.pkl')
        if avgTrainCorrReg < stopCriterion:
            break
#        print(trainHSD_rslt)
        
#        val_rslt = clf.val_loss(feature_test, value_test)
#        #HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#        HSD_rslt = ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#        print(HSD_rslt)

#    val_rslt = clf.val_loss(feature_test, value_test)
##    HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#    HSD_rslt = ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#    print(HSD_rslt)
#    print('\n')
#    recordBest[0][i] = np.sum(val_rslt[1])

    clfBest = Intopt(c, G, A, b, h2, h3, penalty_test, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=6)
    clfBest.model.load_state_dict(torch.load('model.pkl'))
#
    value = clfBest.model(feature_test).squeeze()
    value = value.detach().numpy()
    predValue = np.zeros((value.size, 3))
    
    val_rslt, op = clfBest.val_loss(feature_test, value_test)
    
    for i in range(value.size):
        predValue[i][0] = int(i/day_shift_num)
        predValue[i][1] = value_test[i]
        predValue[i][2] = op[i]
    np.savetxt('./data/2S/2S_penalty' + str(penaltyTerm) + '(' + str(testi) + ').txt', predValue, fmt="%.2f")

    
    #HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
    HSD_rslt = ' avgPReg: ' + str(np.mean(val_rslt))
    print(HSD_rslt)
    recordBest[0][testi] = np.sum(val_rslt)

print(recordBest)
