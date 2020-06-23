# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:36:46 2020

@author: Lenovo
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt


#
#data1 = []
#
#with open('TrainSet1.txt','r') as f:
#    for line in f.readlines():
#        x,y,z = line.strip('\n').split('\t')
#        data1.append([eval(x),eval(y),eval(z)])
#        
#
#data1 = np.array(data1)
#
#dd = [[1,6,1],[1,10,1],[4,11,1],
#      [5,2,-1],[7,6,-1],[10,4,-1]]
#data1 = np.array(dd,dtype=np.float)
#
#
#X = data1[:,:2]
#y = data1[:,-1][:,np.newaxis]
#
#P = np.matmul(X,X.T) * np.matmul(y,y.T)
#m,n = X.shape
#
#G = -np.eye(m)
#A = np.zeros_like(G)
#A = np.squeeze(y)
#b = np.zeros((1))
#h = np.zeros(m)
#q = -np.ones(m)
#
#args = [cvxopt.matrix(P), cvxopt.matrix(q)]
#args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
#args.extend([cvxopt.matrix(A,(1,m)), cvxopt.matrix(b)])
#
#sol = cvxopt.solvers.qp(*args)
#alpha = np.array(sol['x']).reshape((P.shape[1],))
#
#sv_ind = np.where(alpha>0.001)
#sv_alpha = alpha[sv_ind]
#sv = X[sv_ind]
#sv_y = y[sv_ind]
#
#w = np.zeros(n)
#for i in range(len(sv)):
#    w += sv_alpha[i] * sv_y[i] * sv[i]
#
#b = sv_y[0]-np.dot(w,sv[0])
#
#
#x_ = np.linspace(min(X[:,0]),max(X[:,0]),100)
#
#y_ = -(w[0]*x_ + b)/w[1]
#
#plt.plot(x_,y_,'-',label = 'separating hyperplane')
#
#plt.scatter(data1[data1[:,2]==-1][:,0],data1[data1[:,2]==-1][:,1],marker='.',color='r',label='class1',linewidths =0.01)
#plt.scatter(data1[data1[:,2]== 1][:,0],data1[data1[:,2]== 1][:,1],marker='+',color='b',label='class2',linewidths =0.01)
#
#plt.scatter(sv[:,0],sv[:,1],marker='o',color='g',label='support vectors',linewidths =0.01)
#
#plt.xlabel("x")
#plt.ylabel("y")
#plt.legend(loc="lower right")
#plt.title('data')




# 核技巧,第三题

data2 = []

with open('TrainSet2.txt','r') as f:
    for line in f.readlines():
        x,y,z = line.strip('\n').split('\t')
        data2.append([eval(x),eval(y),eval(z)])

data2 = np.array(data2)







X = data2[:,:2]
y = data2[:,-1][:,np.newaxis]
m,n = X.shape



coef0 = 1
degree = 2
kernel_trick_ploy = lambda x1,x2:np.power(1/n * np.dot(x1, x2) + coef0, degree)
kernel_trick_RBF = lambda x1,x2:np.exp(-1/n * np.dot(x1-x2, x1-x2))
kernel_trick = kernel_trick_ploy

K = np.zeros((m,m))
for i in range(m):
    for j in range(m):
        K[i, j] = kernel_trick(X[i], X[j])


P = K*np.matmul(y,y.T)

G = -np.eye(m)
A = np.zeros_like(G)
A = np.squeeze(y)
b = np.zeros((1))
h = np.zeros(m)
q = -np.ones(m)

args = [cvxopt.matrix(P), cvxopt.matrix(q)]
args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
args.extend([cvxopt.matrix(A,(1,m)), cvxopt.matrix(b)])

sol = cvxopt.solvers.qp(*args)
alpha = np.array(sol['x']).reshape((P.shape[1],))

sv_ind = np.where(alpha>0.000001)
sv_alpha = alpha[sv_ind]
sv = X[sv_ind]
sv_y = y[sv_ind]


x_ = np.linspace(min(X[:,0]),max(X[:,0]),501)
y_ = np.linspace(min(X[:,1]),max(X[:,1]),501)

b = sv_y[0].copy()
for i in range(len(sv)):
    b-=sv_alpha[i]*sv_y[i]*kernel_trick(sv[i],sv[0])
print(b)
x_s = []
y_s = []
xp = np.array(x_[0],y_[0])
for i in range(len(x_)):
    for j in range(len(y_)):
        xp = np.array((x_[i],y_[j]))
        x_kernel = np.zeros(len(sv))
        for k in range(len(sv)):
            x_kernel[k] = kernel_trick(sv[k], xp)
        y_pred = np.dot(sv_alpha * np.squeeze(sv_y), x_kernel)+b
        if 0<=y_pred<.01:
            x_s.append(xp[0])
            y_s.append(xp[1])
    


##这是degree=2，coef=0的情况
#W = np.zeros_like((n,n),dtype = np.float64)
#for i in range(len(sv)):
#    W += sv_alpha[i] * sv_y[i] * np.matmul(sv[i].T,sv[i])




#plt.plot(x_s,y_s,'-',label = 'separating hyperplane')

plt.scatter(x_s,y_s,marker='.',color='c',label='separating hyperplane',linewidths =0.01)

plt.scatter(data2[data2[:,2]==-1][:,0],data2[data2[:,2]==-1][:,1],marker='.',color='r',label='class1',linewidths =0.01)
plt.scatter(data2[data2[:,2]== 1][:,0],data2[data2[:,2]== 1][:,1],marker='+',color='b',label='class2',linewidths =0.01)

plt.scatter(sv[:,0],sv[:,1],marker='o',color='g',label='support vectors',linewidths =0.01)

plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="lower right")
plt.title('data')









