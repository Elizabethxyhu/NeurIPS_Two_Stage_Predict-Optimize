# Python program for implementation
# of Ford Fulkerson algorithm
import sys
from collections import defaultdict
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.metrics import mean_squared_error
from numpy import inf

penaltyTerm_list = [0.25, 0.5, 1, 2, 4, 8]
rowSizeG = 4
colSizeG = 10
varNum = colSizeG
#cap = [627.54, 369.72]
cap = [0.8, 60, 40, 2.5]

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
        for i in range(varNum):
            predSol.append(x[i].x)
#        print(sol)
    #        for i in range(allPathNum):
    #            if sol[i] != 0:
    #                print(i, end=" ")
        objective = m.objVal
#        print(predSol, objective)
        
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


testmarkNum = 150

hTemp = np.zeros(testmarkNum*rowSizeG)
for i in range(testmarkNum*rowSizeG):
    hTemp[i] = cap[i%rowSizeG]

startmark = 0
endmark = 10

methodList = ['LR', 'knn5', 'tree', 'rf', 'NN']

for methodName in methodList:
    print(methodName)
    for testmark in range(startmark, endmark):
        cTemp = np.loadtxt('./data/titanium/test_prices/test_prices(' + str(testmark) + ').txt')
        penaltyTemp = np.loadtxt('./data/titanium/test_penalty' + str(penaltyTerm) + '/test_penalty(' + str(testmark) + ').txt')
        GTemp = np.loadtxt('./data/titanium/'+  methodName + '_weights/' + methodName + '_weights(' + str(testmark) + ').txt')
        realGTemp = GTemp[:, 0]
        preGTemp = GTemp[:, 1]

        real_obj = actual_obj(cTemp, realGTemp, hTemp, n_instance=testmarkNum)

        corr_obj_list = []
        for testNum in range(testmarkNum):
    #        print(testNum)
            c_data = np.zeros((colSizeG))
            penalty = np.zeros((colSizeG))
            cntC = testNum * colSizeG
            for i in range(colSizeG):
                c_data[i] = cTemp[cntC]
                penalty[i] = penaltyTemp[cntC]
                cntC = cntC + 1
            c_data = c_data.tolist()
            h_data = np.zeros((rowSizeG))
            cntH = testNum * rowSizeG
            for i in range(rowSizeG):
                h_data[i] = cap[i%rowSizeG]
                cntH = cntH + 1
            h_data = h_data.tolist()

            realG = np.zeros((rowSizeG, colSizeG))
            predG = np.zeros((rowSizeG, colSizeG))
            cnt = testNum * rowSizeG * colSizeG
            for i in range(rowSizeG):
                for j in range(colSizeG):
                    realG[i][j] = realGTemp[cnt]
                    predG[i][j] = preGTemp[cnt]
                    cnt = cnt + 1
            corrrlst = correction_single_obj(c_data, realG, predG, h_data, penalty)
            corr_obj_list.append(corrrlst)

        print("MSE: ", mean_squared_error(realGTemp, preGTemp), "avgCorrReg: ", sum(abs(real_obj - np.array(corr_obj_list)))/testmarkNum, "avgTOV: ", sum(real_obj)/testmarkNum)
    print("\n")

# IntOpt-C
print("IntOpt-C")
for testmark in range(startmark, endmark):
    cTemp = np.loadtxt('./data/titanium/test_prices/test_prices(' + str(testmark) + ').txt')
    penaltyTemp = np.loadtxt('./data/titanium/test_penalty' + str(penaltyTerm) + '/test_penalty(' + str(testmark) + ').txt')
    GTemp = np.loadtxt('./data/titanium/proposed_penalty' + str(penaltyTerm) +'/proposed_penalty('+ str(testmark) + ').txt')
    realGTemp = GTemp[:, 0]
    preGTemp = GTemp[:, 1]

    real_obj = actual_obj(cTemp, realGTemp, hTemp, n_instance=testmarkNum)

    corr_obj_list = []
    for testNum in range(testmarkNum):
#        print(testNum)
        c_data = np.zeros((colSizeG))
        penalty = np.zeros((colSizeG))
        cntC = testNum * colSizeG
        for i in range(colSizeG):
            c_data[i] = cTemp[cntC]
            penalty[i] = penaltyTemp[cntC]
            cntC = cntC + 1
        c_data = c_data.tolist()
        h_data = np.zeros((rowSizeG))
        cntH = testNum * rowSizeG
        for i in range(rowSizeG):
            h_data[i] = cap[i%rowSizeG]
            cntH = cntH + 1
        h_data = h_data.tolist()

        realG = np.zeros((rowSizeG, colSizeG))
        predG = np.zeros((rowSizeG, colSizeG))
        cnt = testNum * rowSizeG * colSizeG
        for i in range(rowSizeG):
            for j in range(colSizeG):
                realG[i][j] = realGTemp[cnt]
                predG[i][j] = preGTemp[cnt]
                cnt = cnt + 1
        corrrlst = correction_single_obj(c_data, realG, predG, h_data, penalty)
#        print("c_data: ", c_data, "h_data: ", h_data, "realG: ", realG, "predG: ", predG)
        corr_obj_list.append(corrrlst)

#    print(corr_obj_list)
    print("MSE: ", mean_squared_error(realGTemp, preGTemp), "avgCorrReg: ", sum(abs(real_obj - np.array(corr_obj_list)))/testmarkNum, "avgTOV: ", sum(real_obj)/testmarkNum)
print("\n")


print('2S')
for penaltyTerm in penaltyTerm_list:
    print(penaltyTerm)
    for testmark in range(startmark, endmark):
        cTemp = np.loadtxt('./data/titanium/test_prices/test_prices(' + str(testmark) + ').txt')
        penaltyTemp = np.loadtxt('./data/titanium/test_penalty' + str(penaltyTerm) + '/test_penalty(' + str(testmark) + ').txt')
        GTemp = np.loadtxt('./data/titanium/2S_weights/2S_weights' + str(penaltyTerm) + '(' + str(testmark) + ').txt')
        realGTemp = GTemp[:, 0]
        preGTemp = GTemp[:, 1]

        real_obj = actual_obj(cTemp, realGTemp, hTemp, n_instance=testmarkNum)

        corr_obj_list = []
        for testNum in range(testmarkNum):
    #        print(testNum)
            c_data = np.zeros((colSizeG))
            penalty = np.zeros((colSizeG))
            cntC = testNum * colSizeG
            for i in range(colSizeG):
                c_data[i] = cTemp[cntC]
                penalty[i] = penaltyTemp[cntC]
                cntC = cntC + 1
            c_data = c_data.tolist()
            h_data = np.zeros((rowSizeG))
            cntH = testNum * rowSizeG
            for i in range(rowSizeG):
                h_data[i] = cap[i%rowSizeG]
                cntH = cntH + 1
            h_data = h_data.tolist()
            meanVal = np.mean(realGTemp)

            realG = np.zeros((rowSizeG, colSizeG))
            predG = np.zeros((rowSizeG, colSizeG))
            meanVal = np.mean(preGTemp)
            cnt = testNum * rowSizeG * colSizeG
            for i in range(rowSizeG):
                for j in range(colSizeG):
                    realG[i][j] = realGTemp[cnt]
                    predG[i][j] = preGTemp[cnt]
                    cnt = cnt + 1
            corrrlst = correction_single_obj(c_data, realG, predG, h_data, penalty)
    #        print("c_data: ", c_data, "h_data: ", h_data, "realG: ", realG, "predG: ", predG)
            corr_obj_list.append(corrrlst)

    #    print(corr_obj_list)
        print("MSE: ", mean_squared_error(realGTemp, preGTemp), "avgCorrReg: ", sum(abs(real_obj - np.array(corr_obj_list)))/testmarkNum, "avgTOV: ", sum(real_obj)/testmarkNum)
    print("\n")

