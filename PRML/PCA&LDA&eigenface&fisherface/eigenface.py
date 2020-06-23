# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:59:50 2020

@author: Lenovo
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as pli

def imresize(original_image,target_size):

    i = Image.fromarray(original_image)
    ii = i.resize(target_size[::-1],Image.BILINEAR)
    image = np.asarray(ii)
    return image

def downsampling(data,stride):
    return data[::stride,::stride]

def Convolve(I):
    F = np.array([[1/9]*9]).reshape(3,3)
    iw,ih = I.shape
    fw,fh = F.shape
    Conv_I = np.zeros_like(I)
    Image_pad = np.pad(I,(((fh-1)//2,(fh-1)//2),((fw-1)//2,(fw-1)//2)), 'reflect')
    print(Image_pad.shape)
    func = lambda x,y:np.sum(Image_pad[x:x+fh,y:y+fw]*F)
    for i in range(iw):
        for j in range(ih):
            Conv_I[i,j] = func(i,j)
    return Conv_I

def PCA_info(data):
    samples,dimensions = data.shape
    m = data.mean(axis=0)
    z = (data-m).astype(np.float32)
    S = np.matmul(z.T,z)/samples
    
    eigvals,eigvec=np.linalg.eigh(S)
    
    eigvec = eigvec[:,eigvals.argsort()[::-1]]
    eigvals = eigvals[eigvals.argsort()[::-1]]
    return m,eigvec,eigvals
    

def PCA(data,eigvec,m,n_components):
    
    '''输入数据，特征向量，均值，降的维度——输出降维后的特征'''
    M = eigvec[:,:n_components]
    z = (data-m).astype(np.float32)
    return np.matmul(z,M)


def get_data_set(downsampling = 8):
    Subjects = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    file_list = os.listdir('.\\project1_data_Recognition')
    for pics_name in file_list:
        
        if pics_name[-3:] != 'pgm':
            continue
        pic = pli.imread(os.path.join('.\\project1_data_Recognition',pics_name))
        pic = pic[::downsampling,::downsampling]
        Subjects[int(pics_name[7:9])-1].append(pic)
        
    return Subjects


def get_train_test_data(Subjects,N=3):
    
    
    randperm = np.random.permutation(np.arange(11))
    train_ind = randperm[:N]
    test_ind = randperm[N:]
    
    if type(Subjects) == list:
        Subs = np.array(Subjects)
    Subs = Subjects.copy()
  
    shape = Subs.shape
    
    train_set =  Subs[:,train_ind,:,:].reshape(15*N,shape[2]*shape[3])
    test_set = Subs[:,test_ind,:,:].reshape(15*(11-N),shape[2]*shape[3])
    
    return train_set,test_set


#分类,计算错误率
def classify_accuracy(redu_data,test_set,N):
    close_ind = []
    wrong = 0
    for i in range(len(test_set)):
        test = test_set[i]
        
        distances = np.square(redu_data-test).sum(axis=1)
        min_dis_ind = np.argmin(distances)
        
        close_ind.append(min_dis_ind//N)
        if min_dis_ind//N != i//(11-N):
            wrong+=1
        
    misrate = wrong/len(test_set)

    return close_ind,misrate

def exc_experiment_PCA(Subjects,N,K=None):
    
    
    train_set,test_set = get_train_test_data(Subjects,N)
    print(train_set.shape,test_set.shape)
    m,eigvec,eigvals = PCA_info(train_set)
    
    
    if K is None:
        K = np.arange(0,100,10)
    
    misrates = []
    for n_components in K:
        redu_train = PCA(train_set,eigvec,m,n_components)
        redu_test =  PCA(test_set,eigvec,m,n_components)
        close_ind,misrate = classify_accuracy(redu_train,redu_test,N)
        
        misrates.append(misrate)
    
    return np.array(misrates),m


def exc_experiment_PCA_10_times(Subjects,N,K=None):
    
    misrates = np.zeros(len(K),dtype=np.float32)
    for _ in range(10):
        misrate,_ = exc_experiment_PCA(Subjects,N,K)
        misrates += misrate
    
    
    return misrates/10


def get_eigen(train_set,test_set,N):
    m,eigvec,eigvals = PCA_info(train_set)
    
    redu_train = PCA(train_set,eigvec,m,15*N-15)
    redu_test =  PCA(test_set,eigvec,m,15*N-15)
    
    mi = redu_train.reshape(N,15,15*N-15)
    mi = mi.mean(axis=0)
    m = redu_train.mean(axis=0)
    
    
    Sw = np.zeros((15*N-15,15*N-15),dtype=np.float64)
    for i in range(15):
        Xi = redu_train[i*N:(i+1)*N] - mi[i]
        Si = np.matmul(Xi.T,Xi).astype(np.float64)
        Sw+=Si
    
    
    ma = mi-m
    Sb = 15*N*np.matmul(ma.T,ma)    
    

    Mat = np.matmul(np.linalg.inv(Sw),Sb)
    eigvals,eigvec = np.linalg.eig(Mat)
    eigvec = eigvec[:,eigvals.argsort()[::-1]]
    
    return redu_train,redu_test,eigvec
    


def exc_experiment_LDA_(Subjects,N,K):
    train_set,test_set = get_train_test_data(Subjects,N)
    redu_train,redu_test,eigvec = get_eigen(train_set,test_set,N)
    
    print(redu_train.shape)
    misrates = []
    K = min(K,14)
    for n_components in range(1,K):
        M = eigvec[:,:n_components]
        train_features = np.matmul(redu_train,M)
        test_features = np.matmul(redu_test,M)
        close_ind,misrate = classify_accuracy(train_features,test_features,N)
        
        misrates.append(misrate)

    return np.array(misrates)


def exc_experiment_LDA(Subjects,N,K):    
    train_set,test_set = get_train_test_data(Subjects,N)
    m,eigvec,eigvals = PCA_info(train_set)
    
    redu_train = PCA(train_set,eigvec,m,15*N-15)
    redu_test =  PCA(test_set,eigvec,m,15*N-15)
    
    mi = redu_train.reshape(15,N,15*N-15)
    mi = mi.mean(axis=1)
    m = redu_train.mean(axis=0)
    
    ma = mi-m
    
    Sb = N*np.matmul(ma.T,ma)
    
    Sw = np.zeros((15*N-15,15*N-15),dtype=np.float64)
    for i in range(15):
        Xi = redu_train[i*N:(i+1)*N] - mi[i]
        Si = np.matmul(Xi.T,Xi).astype(np.float64)
        Sw+=Si
    
    
    Mat = np.matmul(np.linalg.inv(Sw),Sb)
    eigvals,eigvec = np.linalg.eig(Mat)
    
    eigvec = eigvec[:,eigvals.argsort()[::-1]]
    
    #获得训练集降维特征
    
    
    
    misrates = []
    for n_components in range(1,K+1):
        M = eigvec[:,:n_components]
        train_features = np.matmul(redu_train,M)
        test_features = np.matmul(redu_test,M)        

        close_ind,misrate = classify_accuracy(train_features,test_features,N)
        misrates.append(misrate)
    
    return np.array(misrates)


def exc_experiment_LDA_10_times(Subjects,N,K):
    

    misrates = np.zeros(K,dtype=np.float32)
    print(misrates.shape)
    for _ in range(10):
        misrates += exc_experiment_LDA(Subjects,N,K)
    
    
    return misrates/10




#Subjects = get_data_set(downsampling = 4)

#Ks = np.arange(5,100,5)
#misrate1 = exc_experiment_PCA_10_times(Subjects,3,Ks)
#misrate2 = exc_experiment_PCA_10_times(Subjects,5,Ks)
#misrate3 = exc_experiment_PCA_10_times(Subjects,7,Ks)
#plt.plot(Ks,misrate1,'+-',label = 'N=3')
#plt.plot(Ks,misrate2,'+-',label = 'N=5')
#plt.plot(Ks,misrate3,'+-',label = 'N=7')


#plt.xlabel("K(feature dimensions)")
#plt.ylabel("misclassification rate")
#plt.legend(loc="upper right")
#plt.title('the average misclassiﬁcation rate vs. the dimensions of Eigenspace')



#LDA

Ks = np.arange(1,15)
misrate1 = exc_experiment_LDA_10_times(Subjects,3,14)
misrate2 = exc_experiment_LDA_10_times(Subjects,5,14)
misrate3 = exc_experiment_LDA_10_times(Subjects,7,14)
plt.plot(Ks,misrate1,'+-',label = 'N=3')
plt.plot(Ks,misrate2,'+-',label = 'N=5')
plt.plot(Ks,misrate3,'+-',label = 'N=7')


plt.xlabel("K(feature dimensions)")
plt.ylabel("misclassification rate")
plt.legend(loc="upper right")
plt.title('the average misclassiﬁcation rate vs. K')





        