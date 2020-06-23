# -*- coding: utf-8 -*-
"""
Created on Sat May  9 00:44:38 2020

@author: Lenovo
"""



import numpy as np
import matplotlib.pyplot as plt
import re


def plot(Y1,Y2,Y3,method = 'PCA'):


    plt.scatter(Y1,[1]*len(Y1),marker='+',color='r',label='class1',linewidths =0.01)
    plt.scatter(Y2,[1]*len(Y2),marker='.',color='b',label='class2',linewidths =0.01)
    plt.scatter(Y3,[1]*len(Y3),marker='.',color='g',label='class3',linewidths =0.01)
    
    plt.xlabel("x")
    plt.legend(loc="center left")
    plt.title(' The ﬁrst 1000 samples of each category after reduction(%s)'%method)
    
    
    plt.yticks([])
    plt.ylim(0,10)
    
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.show()


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
        x,y = re.findall(r'([0-9-].*)  ([0-9-].*)',line)[0]
        data3.append([eval(x),eval(y)])

data3 = np.array(data3)

#
#plt.scatter(data1[:1000,0],data1[:1000,1],marker='.',color='r',label='class1',linewidths =0.01)
#plt.scatter(data2[:1000,0],data2[:1000,1],marker='+',color='b',label='class2',linewidths =0.01)
#plt.scatter(data3[:1000,0],data3[:1000,1],marker='v',color='g',label='class3',linewidths =0.01)
#plt.xlabel("x")
#plt.ylabel("y")
#plt.legend(loc="upper left")
#plt.title(' The ﬁrst 1000 samples of each category')

def PCA_2d(data):
    m = data.mean(axis=0)
    z = data-m
    S = np.zeros((2,2))
    for zi in z:
        S += zi[np.newaxis,:].T.dot(zi[np.newaxis,:])
    print(S)
    
    eigvals,eigvec=np.linalg.eig(S)
    M = eigvec[:,np.argmax(eigvals)]
    

    return M,m
def get_coord(data,m,M):
    z = data - m
    Y = (M*z).sum(axis=1)
    return Y
def PCA(d1,d2,d3,plo = True):
    data = np.concatenate((d1,d2,d3),axis=0)
    M,m = PCA_2d(data)
    Y1 = get_coord(d1,m,M)
    Y2 = get_coord(d2,m,M)
    Y3 = get_coord(d3,m,M)
    
    if plo:
        plot(Y1,Y2,Y3,'PCA')
    return Y1,Y2,Y3,M,m
        






def FLD(d1,d2,d3,plo=True):
    data = np.concatenate((d1,d2,d3),axis=0)

    Sb = np.zeros((2,2))
    mu = data.mean(axis=0)
    mui = [d1.mean(axis=0),d2.mean(axis=0),d3.mean(axis=0)]
    
    for mi in mui:
        m = mi - mu
        Sb += 1000*m[:,np.newaxis].dot(m[:,np.newaxis].T)
    
    
    Sw = np.zeros((2,2))
    
    d = [d1,d2,d3]
    for c in range(3):
        ds = d[c] - mui[c]
        for X in ds:
            Sw+=X[:,np.newaxis].dot(X[:,np.newaxis].T)
    
    Mat = np.matmul(np.linalg.inv(Sw),Sb)
    eig,eigv = np.linalg.eig(Mat)
    
    M = eigv[:,np.argmax(eig)]
    
    Y1 = (M*d1).sum(axis=1)
    Y2 = (M*d2).sum(axis=1)
    Y3 = (M*d3).sum(axis=1)
    if plo:
        plot(Y1,Y2,Y3,'FLD')
    
    return Y1,Y2,Y3,M,mu
    

def acc(Y1,Y2,Y3,T1,T2,T3,k=5):

    
    class1 = np.concatenate((Y1[:,np.newaxis],Y1[:,np.newaxis]),axis = 1)
    class1[:,1]=1
    class2 = np.concatenate((Y2[:,np.newaxis],Y2[:,np.newaxis]),axis = 1)
    class2[:,1]=2
    class3 = np.concatenate((Y3[:,np.newaxis],Y3[:,np.newaxis]),axis = 1)
    class3[:,1]=3
    
    clas = np.concatenate((class1,class2,class3),axis = 0)
    
    Test1 = np.concatenate((T1[:,np.newaxis],T1[:,np.newaxis]),axis = 1)
    Test1[:,1]=1
    Test2 = np.concatenate((T2[:,np.newaxis],T2[:,np.newaxis]),axis = 1)
    Test2[:,1]=2
    Test3 = np.concatenate((T3[:,np.newaxis],T3[:,np.newaxis]),axis = 1)
    Test3[:,1]=3
    
    Tests = [Test1,Test2,Test3]
    
    right= [0.]*3
    vote = [0.]*3
    for i in range(3):
        Test = Tests[i]
        for X in Test:
            vote = [0]*3
            sta = clas-X
            sta[:,0] = np.abs(sta[:,0])
            
            sta = sta[np.argsort(sta[:,0])][0:k]
            
            for j in sta:

                vote[int(j[1])]+=1
            if np.argmax(vote) == 0:
                right[i]+=1
                if vote[1] == vote[0] or vote[2] == vote[0]:
                    if vote[1]==vote[2]==vote[0]:
                        right[i]-=2/3
                    else:
                        right[i]-=0.5
    
    mis = [1-r/len(Test1) for r in right]
    print('mis rate:',mis)

    return mis




def plot_acc(Y1,Y2,Y3,T1,T2,T3,n=7):
    
    class1 = np.concatenate((Y1[:,np.newaxis],Y1[:,np.newaxis]),axis = 1)
    class1[:,1]=1
    class2 = np.concatenate((Y2[:,np.newaxis],Y2[:,np.newaxis]),axis = 1)
    class2[:,1]=2
    class3 = np.concatenate((Y3[:,np.newaxis],Y3[:,np.newaxis]),axis = 1)
    class3[:,1]=3
    
    clas = np.concatenate((class1,class2,class3),axis = 0)
    
    Test1 = np.concatenate((T1[:,np.newaxis],T1[:,np.newaxis]),axis = 1)
    Test1[:,1]=1
    Test2 = np.concatenate((T2[:,np.newaxis],T2[:,np.newaxis]),axis = 1)
    Test2[:,1]=2
    Test3 = np.concatenate((T3[:,np.newaxis],T3[:,np.newaxis]),axis = 1)
    Test3[:,1]=3
    
    Tests = [Test1,Test2,Test3]
    
    right= [0]*n
    vote = [0]*3
    
    for i in range(3):
        Test = Tests[i]
        for X in Test:
            vote = [0]*3
            sta = clas-X
            sta[:,0] = np.abs(sta[:,0])

            sta = sta[np.argsort(sta[:,0])][0:n]
            
            for j in range(len(sta)):
                vote[int(sta[j][1])]+=1
                if np.argmax(vote) == 0:
                    right[j]+=1
                    if vote[1] == vote[0] or vote[2] == vote[0]:
                        if vote[1]==vote[2]==vote[0]:
                            right[j]-=2/3
                        else:
                            right[j]-=0.5                
    
    accs = [1-r/len(clas) for r in right]

    return accs



n = 50


d1,d2,d3 = data1[:1000],data2[:1000],data3[:1000]

data = np.concatenate((d1,d2,d3),axis=0)
Y1,Y2,Y3,M,m= PCA(d1,d2,d3,0)

#plot(Y1,Y2,Y3)
dt1,dt2,dt3 = data1[1000:],data2[1000:],data3[1000:]
T1 = (M*(dt1-m)).sum(axis=1)
T2 = (M*(dt2-m)).sum(axis=1)
T3 = (M*(dt3-m)).sum(axis=1)


print('PCA')
ac1 = acc(Y1,Y2,Y3,T1,T2,T3,5)

accs_PCA = plot_acc(Y1,Y2,Y3,T1,T2,T3,n)


Y1,Y2,Y3,M,m = FLD(d1,d2,d3,0)
T1 = (M*dt1).sum(axis=1)
T2 = (M*dt2).sum(axis=1)
T3 = (M*dt3).sum(axis=1)
#plot(Y1,Y2,Y3,'FLD')

accs_FLD = plot_acc(Y1,Y2,Y3,T1,T2,T3,n)
print('FLD')
ac2 = acc(Y1,Y2,Y3,T1,T2,T3,5)



##
plt.plot(list(range(1,n+1)),accs_PCA,label='PCA')
plt.title(' misclassiﬁcation rate of k nearest-neighbors classiﬁer')
plt.xlabel('k')
plt.ylabel('misclassiﬁcation rate')

plt.plot(list(range(1,n+1)),accs_FLD,label='FLD')
plt.title(' misclassiﬁcation rate of k nearest-neighbors classiﬁer')
plt.xlabel('k')
plt.ylabel('misclassiﬁcation rate')
plt.legend(loc='upper right')
#
