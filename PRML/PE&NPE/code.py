# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:59:24 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



data1 = []
data2 = []
data3 = []

with open('data1.txt','r') as f:
    for line in f.readlines():
        x,y = re.findall(r'([0-9-].*)  ([0-9-].*)',line)[0]
        data1.append([eval(x),eval(y)])

data1 = np.array(data1)

with open('data2.txt','r') as f:
    for line in f.readlines():
        x,y = re.findall(r'([0-9-].*)  ([0-9-].*)',line)[0]
        data2.append([eval(x),eval(y)])

data2 = np.array(data2)
 
with open('data3.txt','r') as f:
    for line in f.readlines():
        x,y = re.findall(r'(".*").*(".*").*(".*")',str(aa))
        data3.append([eval(x),eval(y)])

data3 = np.array(data3)

#
#plt.scatter(data1[:1000,0],data1[:1000,1],marker='.',color='r',label='class1',linewidths =0.01)
#plt.scatter(data2[:1000,0],data2[:1000,1],marker='+',color='b',label='class2',linewidths =0.01)
#plt.scatter(data3[:1000,0],data3[:1000,1],marker='v',color='g',label='class3',linewidths =0.01)
#plt.xlabel("x")
#plt.ylabel("y")
#plt.legend(loc="lower right")
#plt.title('The ﬁrst 1000 samples of each category')

def get_normal_theta(data):
    m = data.mean(axis=0)
    return m,np.matmul((data-m).T,(data-m))/len(data)




def class_conditional_proba(X,mu,sigma):
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)    

    Z = 1/(2*np.pi*det**0.5)*np.e**(-0.5*np.matmul(X-mu,inv).dot((X-mu).T))
    
    return Z
    
def class_conditional_density(X,Y,mu,sigma):
    Z = np.zeros_like(X)

    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i,j],Y[i,j]])
            Z[i,j] = 1/(2*np.pi*det**0.5)*np.e**(-0.5*np.matmul(x-mu,inv).dot((x-mu).T))
    return Z

def plot_contour(mu,sigma,i):
    x = np.linspace(-10,10,1000)
    y = np.linspace(-10,10,1000)
    X, Y = np.meshgrid(x, y)
    
    Z = class_conditional_density(X,Y,mu,sigma)
    
    plt.contour(X,Y,Z,label = 'class%d'%i )
    
def MAP_classifer(X,mu,sigma):
    mu1,mu2,mu3 = mu
    sigma1,sigma2,sigma3 = sigma
    
    
    p1 = class_conditional_proba(X,mu1,sigma1)
    p2 = class_conditional_proba(X,mu2,sigma2)
    p3 = class_conditional_proba(X,mu3,sigma3)
    
    if p1>p2 and p1>p3:
        return 0
    elif p2>p1 and p2>p3:
        return 1
    else:
        return 2

#trainnum = 500
#train1 = data1[:trainnum]
#train2 = data2[:trainnum]
#train3 = data3[:trainnum]

#mu1,sigma1 = get_normal_theta(train1)
#mu2,sigma2 = get_normal_theta(train2)
#mu3,sigma3 = get_normal_theta(train3)
##
#plot_contour(mu1,sigma1,1)
#plot_contour(mu2,sigma2,2)
#plot_contour(mu3,sigma3,3)
#
#
#plt.xlabel("x")
#plt.ylabel("y")
#mu = [mu1,mu2,mu3]
#sigma = [sigma1,sigma2,sigma3]
#
#Confusion_m = np.zeros((3,3))
#for X in data1[trainnum:]:
#    c= MAP_classifer(X,mu,sigma)
#    Confusion_m[0,c]+=1
#    
#for X in data2[trainnum:]:
#    c= MAP_classifer(X,mu,sigma)
#    Confusion_m[1,c]+=1
#    
#for X in data3[trainnum:]:
#    c= MAP_classifer(X,mu,sigma)
#    Confusion_m[2,c]+=1




trainnum = 1000
train1 = data1[:trainnum]
train2 = data2[:trainnum]
train3 = data3[:trainnum]


#x = np.linspace(-10,10,100)
#y = np.linspace(-10,10,100)
#X, Y = np.meshgrid(x, y)
#
#Z = np.zeros_like(X)
#for i in range(Z.shape[0]):
#    for j in range(Z.shape[1]):
#        x = np.array([X[i,j],Y[i,j]])
#        T = np.sum((train1 - x)**2,axis = 1)
#        p = np.sum(1/(2*np.pi)*np.e**(-0.5*T))
#        Z[i,j] = p
#
#Z = np.zeros_like(X)
#for i in range(Z.shape[0]):
#    for j in range(Z.shape[1]):
#        x = np.array([X[i,j],Y[i,j]])
#        T = np.sum((train2 - x)**2,axis = 1)
#        p = np.sum(1/(2*np.pi)*np.e**(-0.5*T))
#        Z[i,j] = p
#
#Z = np.zeros_like(X)
#for i in range(Z.shape[0]):
#    for j in range(Z.shape[1]):
#        x = np.array([X[i,j],Y[i,j]])
#        T = np.sum((train3 - x)**2,axis = 1)
#        p = np.sum(1/(2*np.pi)*np.e**(-0.5*T))
#        Z[i,j] = p

#def MAP_NP(X,train1,train2,train3):
#    
#    T = np.sum((train1 - X)**2,axis = 1)
#    p1 = np.sum(1/(2*np.pi)*np.e**(-0.5*T))
#    
#    T = np.sum((train2 - X)**2,axis = 1)
#    p2 = np.sum(1/(2*np.pi)*np.e**(-0.5*T))
#    
#    T = np.sum((train3 - X)**2,axis = 1)
#    p3 = np.sum(1/(2*np.pi)*np.e**(-0.5*T))    
#    
#    if p1>p2 and p1>p3:
#        return 0
#    elif p2>p3 and p2>p1:
#        return 1
#    else:
#        return 2
#    
#
#
#Confusion_m = np.zeros((3,3))
#for X in data1[trainnum:]:
#    c= MAP_NP(X,train1,train2,train3)
#    Confusion_m[0,c]+=1
#    
#for X in data2[trainnum:]:
#    c= MAP_NP(X,train1,train2,train3)
#    Confusion_m[1,c]+=1
#    
#for X in data3[trainnum:]:
#    c= MAP_NP(X,train1,train2,train3)
#    Confusion_m[2,c]+=1
#
#rate = Confusion_m/trainnum
#
#misclassification_rate = 1 - rate[[0,1,2],[0,1,2]]


def NN(X,train1,train2,train3):
    
    data = np.concatenate((train1,train2,train3),axis = 0)
    
    data = np.sum((data-X)**2,axis = 1)
    clas = np.argmin(data)//1000
    return int(clas)
    
def kNN(X,train1,train2,train3,k=10):
    data = np.concatenate((train1,train2,train3),axis = 0)
    distance = np.sum((data-X)**2,axis = 1)
    disrank = np.argsort(distance)
    count = np.zeros(3)
    for i in range(len(disrank)):
        if(disrank[i]<k):
            count[int(i//1000)]+=1
    
    return np.argmax(count)
    

Confusion_m = np.zeros((3,3))
for X in data1[trainnum:]:
    d = np.concatenate((train1,train2,train3),axis = 0)
    clas = np.zeros(3000)
    clas[:1000] = 0
    clas[1000:2000] = 1
    clas[2000:3000] = 2
    
    clas = clas[:,np.newaxis]
    distance = np.sum((d-X)**2,axis = 1)
    data = np.concatenate((distance[:,np.newaxis],clas),axis = 1)
    disrank = np.argsort(distance)
    
    data = data[disrank][:10]

    
    count = np.array([0,0,0],dtype=np.float32)
    for dd in data:
        count[int(dd[1])]+=1
    
    c = np.argmax(count)
    
    Confusion_m[0,c]+=1
    
    if count[c] == count[c-1]:
        Confusion_m[0,c]-=.5
        Confusion_m[0,c-1]+=.5
    if count[c] == count[c-2]:
        Confusion_m[0,c]-=.5
        Confusion_m[0,c-2]+=.5
    
for X in data2[trainnum:]:
    d = np.concatenate((train1,train2,train3),axis = 0)
    clas = np.zeros(3000)
    clas[:1000] = 0
    clas[1000:2000] = 1
    clas[2000:3000] = 2
    
    clas = clas[:,np.newaxis]
    distance = np.sum((d-X)**2,axis = 1)
    data = np.concatenate((distance[:,np.newaxis],clas),axis = 1)
    disrank = np.argsort(distance)
    
    data = data[disrank][:10]

    
    count = np.array([0,0,0],dtype=np.float32)
    for dd in data:
        count[int(dd[1])]+=1
    
    c = np.argmax(count)
    
    Confusion_m[1,c]+=1
    
    if count[c] == count[c-1]:
        Confusion_m[1,c]-=.5
        Confusion_m[1,c-1]+=.5
    if count[c] == count[c-2]:
        Confusion_m[1,c]-=.5
        Confusion_m[1,c-2]+=.5

    
    
    
    
for X in data3[trainnum:]:
    
    d = np.concatenate((train1,train2,train3),axis = 0)
    clas = np.zeros(3000)
    clas[:1000] = 0
    clas[1000:2000] = 1
    clas[2000:3000] = 2
    
    clas = clas[:,np.newaxis]
    distance = np.sum((d-X)**2,axis = 1)
    data = np.concatenate((distance[:,np.newaxis],clas),axis = 1)
    disrank = np.argsort(distance)
    
    data = data[disrank][:10]

    
    count = np.array([0,0,0],dtype=np.float32)
    for dd in data:
        count[int(dd[1])]+=1
    
    c = np.argmax(count)
    
    Confusion_m[2,c]+=1
    
    if count[c] == count[c-1]:
        Confusion_m[2,c]-=.5
        Confusion_m[2,c-1]+=.5
    if count[c] == count[c-2]:
        Confusion_m[2,c]-=.5
        Confusion_m[2,c-2]+=.5

rate = Confusion_m/trainnum

misclassification_rate = 1 - rate[[0,1,2],[0,1,2]]
















