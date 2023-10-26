# Python program for implementation
# of Ford Fulkerson algorithm
import sys
from collections import defaultdict
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.metrics import mean_squared_error
from numpy import inf
from ip_model_whole import solve_LP
import torch

nurse_num = 15
day_num = 7
shift_num = 3
#serve_patient_num = [17,8,14,25,19,11,5,16,13,27,11,28,27,14,6,22]
#serve_patient_num = 20
decision_num = nurse_num * day_num * shift_num
day_shift_num = day_num * shift_num
testmarkNum = 90
penaltyTerm = 4

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
#    G1_serve = np.zeros(decision_num)
#    for i in range(nurse_num):
#        for j in range(day_shift_num):
#            G1_serve[i*day_shift_num+j] = serve_patient_num[i]
#    print(G1_serve)
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
        sol = []
        try:
            for i in range(decision_num):
                sol.append(x[i].x)
            objective = m.objVal
        except:
            for i in range(decision_num):
                sol.append(0)
            objective = 0

        obj_list.append(objective)
        
#        print("Sol: ",sol)
        
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
    
#    np.savetxt('c.txt', c, fmt="%.2f")
#    np.savetxt('predSol.txt', predSol, fmt="%.2f")
#    np.savetxt('Sol.txt', sol, fmt="%.2f")
##    print("c: ", c)
##    print("preSol: ", predSol)
#    print("preSol: ", sol)
#    trueObj = 0
#    for i in range(decision_num):
#        trueObj = trueObj + c[i] * sol[i]
#    print("trueObj: ", trueObj)
#    print("Obj: ", objective)
    
    return objective
      

startmark = int(sys.argv[1])
endmark = startmark + 10
#methodList = ['NN', 'ridge', 'knn5', 'tree', 'rf']
methodList = ['2S']
testList = [5,8]

for methodName in methodList:
    print(methodName)
#    for testmark in range(startmark, endmark):
    for testmark in testList:
        perference = np.loadtxt('./data/preference/preference(' + str(testmark) + ').txt')
        c = np.zeros(decision_num)
        for i in range(nurse_num):
            for j in range(day_shift_num):
                c[i*day_shift_num+j] = perference[i][j]
        serve_patient_num = np.loadtxt('./data/serve_patient_num/serve_patient_num(' + str(testmark) + ').txt')
#        patient_num_temp = np.loadtxt('./data/' + methodName + '/' + methodName + '(' + str(testmark) + ').txt')
        patient_num_temp = np.loadtxt('./data/' + methodName + '/' + methodName + '_penalty' + str(penaltyTerm) + '(' + str(testmark) + ').txt')
        penalty_test = np.loadtxt('./data/penalty' + str(penaltyTerm) + '/test_penalty' + str(penaltyTerm) + '/test_penalty(' + str(testmark) + ').txt')
        real_patient_num = patient_num_temp[:,1]
        pre_patient_num = patient_num_temp[:,2]

        A,b,G,h2,h3 = gen_matrix(nurse_num,day_num,shift_num,serve_patient_num,decision_num,day_shift_num)
        #print(G,h)
        real_obj = actual_obj(c, A, b, G, real_patient_num, h2, h3, n_instance=testmarkNum)
#        print(real_obj)
        corr_obj_list = []
        ReLUValue = np.mean(real_patient_num)
        for testNum in range(testmarkNum):
        #        print(testNum)
            real_patient = {}
            pre_patient = {}
            for i in range(day_shift_num):
                real_patient[i] = real_patient_num[i+testNum*day_shift_num]
#                pre_patient[i] = ReLUValue
                pre_patient[i] = pre_patient_num[i+testNum*day_shift_num]

            penalty = np.zeros(decision_num)
            for j in range(decision_num):
                penalty[j] = penalty_test[j+testNum*decision_num]

            corrrlst = correction_single_obj(c, A, b, G, real_patient, pre_patient, h2, h3, penalty)
            corr_obj_list.append(corrrlst)

#        print(corr_obj_list)
        print("corrReg: ", sum(abs(real_obj - np.array(corr_obj_list)))/testmarkNum, "TOV: ", sum(real_obj)/testmarkNum)

