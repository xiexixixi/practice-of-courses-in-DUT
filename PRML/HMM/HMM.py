# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:49:10 2020

@author: Lenovo
"""
#from DTW import * 
import DTW
from K_Means_pp import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pickle
import re
import os

def get_sample_list(vowel):
    file_path = './project2_data/%s'%vowel
    with open(file_path) as f:
        xml = f.read()
        soup = BeautifulSoup(xml,'xml')
        sample_list = soup.find_all('trainingExample')
        
        s_lis = []
        for sample in sample_list:
            coord_list = sample.find_all('coord')
            
            c_lis = []
            for coord in coord_list:
                t,x,y = [eval(n) for n in re.findall(r'(".*").*(".*").*(".*")',str(coord))[0]]
                t,x,y = eval(t),eval(x),eval(y)
                c_lis.append([float(t),x,y])
    
            s_lis.append(c_lis)
    return s_lis

def get_vowel_list():
    file_list = os.listdir('.\\project2_data')
    vowel_lis = []
    for vowel in file_list:
        vowel_lis.append(get_sample_list(vowel))
    
    return vowel_lis

def get_df_Vowels():
    df_Vowels = pd.DataFrame(columns = ['A','E','I','O','U'],index = ['train','test'])
    vowel_lis =  get_vowel_list()
    for i in range(5):
        vc = ['A','E','I','O','U'][i]
        v = vowel_lis[i]
    
        df_Vowels[vc]['train'] = v[1::2]
        df_Vowels[vc]['test'] = v[::2]
        
    df_Vowels.to_pickle(r'./data_pickle/df_Vowels_ori.pkl')
    return
                

def get_train_coord(number = None):
    with open(r'./data_pickle/df_Vowels_ori.pkl','rb') as f:
        df_Vowels = pickle.load(f)
    #得到聚类坐标
    all_train_coord = []
    if number is None:
        for v in df_Vowels.columns:
            samples = df_Vowels[v]['train'].copy()
            for s in samples:
                for coord in s:
                    all_train_coord.append(coord[1:])
    else:

        for v in df_Vowels.columns:
            i = 0
            samples = df_Vowels[v]['train'].copy()
            for s in samples:
                i+=1
                if i> number:
                    break
                for coord in s:
                    all_train_coord.append(coord[1:])        
    return all_train_coord


#==========================================================================

class HMM(object):
    def __init__(self,sn,vn,A=None,B=None,pi=None,name = None,model = 'left_right'):
        
        self.sn = sn
        self.vn = vn
        self.name = name

        self.A = A
        self.B = B
        self.pi = pi
        
        self.learn_by = None
        
        self.c = None
        self.alpha = None
        self.beta = None
        
        self.xi = None
        self.gamma = None
        self.delta = None
    
        self.error = True
        self.dtype = np.float32
        
        self.model = model
    def forward(self,observ_seq):
        
        K = len(observ_seq)
        self.alpha = np.zeros((K,self.sn),dtype=self.dtype)
#        print(self.pi,self.B[:,observ_seq[0]])
        
        self.c = np.zeros(K,dtype = np.float32)
        self.alpha[0] = self.pi*self.B[:,observ_seq[0]]
        if self.alpha[0].sum() == 0:
            return self.alpha
#            self.alpha[0]+=1e-20     
        self.c[0] = 1/self.alpha[0].sum()
        self.alpha[0] = self.alpha[0]*self.c[0]
            
        for k in range(1,K):
            for i in range(self.sn):
                s = 0.
                for j in range(self.sn):
                    s += self.alpha[k-1,j] * self.A[j,i]
                self.alpha[k,i] = s*self.B[i,observ_seq[k]]
                
            if self.alpha[k].sum() == 0:
                return self.alpha
#                self.alpha[k]+=1e-20

            self.c[k] = 1/self.alpha[k].sum()
            self.alpha[k] = self.alpha[k]*self.c[k]
            
        return self.alpha
    
    def backward(self,observ_seq):
        K = len(observ_seq)
        self.beta = np.zeros((K,self.sn),dtype=self.dtype)
        self.beta[-1,:] = self.c[-1]
        
        for k in range(K-2,-1,-1):
            for i in range(self.sn):
                s=0
                for j in range(self.sn):
                    s += self.beta[k+1,j]*self.A[i,j]*self.B[j,observ_seq[k+1]]
                self.beta[k,i] = s*self.c[k]
        
        return self.beta
        
    def _update_xi(self):
        observ_seq = self.learn_by
        
        K = len(observ_seq)
        self.xi = np.zeros((K-1,self.sn,self.sn),dtype=self.dtype)
        for k in range(K-1):
            s = 0
            for i in range(self.sn):
                ss = 0
                for j in range(self.sn):
                    ss+=self.alpha[k,i]*self.A[i,j]*self.B[j,observ_seq[k+1]]*self.beta[k+1,j]
                s+=ss
            if s:
                for i in range(self.sn):
                    for j in range(self.sn):
                        self.xi[k,i,j] = self.alpha[k,i]*self.A[i,j]*self.B[j,observ_seq[k+1]]*self.beta[k+1,j]/s
            else:
                return False
        return True
    
    def _update_gamma(self):
        
        observ_seq = self.learn_by
        K = len(observ_seq)
        self.gamma = np.zeros((K,self.sn),self.dtype)
        for k in range(K):
            s = np.dot(self.alpha[k],self.beta[k])
            if s:
                self.gamma[k] = (self.alpha[k] * self.beta[k])/s
            else:
                return False
                
        return True
    
    
    
    def _init_para(self,A=None,B=None,pi=None):
        
        
        if self.model == 'left_right':
            r = np.zeros((self.sn,self.sn),dtype = self.dtype)
            for i in range(self.sn-1):
#                r[[i,i],[i,(i+1)%self.sn]] = 0.5
                r[i,i] = np.random.rand()
                r[i,i+1] = 1-r[i,i]
            r[-1,-1] = 1
            self.A = r if A is None else A
            
            r = np.random.random((self.sn,self.vn))
            self.B = (r.T/r.sum(axis=1)).T if B is None else B
            
            self.pi = np.zeros(self.sn,dtype = self.dtype)
            self.pi[0] = 1
        
        else:
            r = np.random.random((self.sn,self.sn))
            self.A = (r.T/r.sum(axis=1)).T if A is None else A
            
            r = np.random.random((self.sn,self.vn))
            self.B = (r.T/r.sum(axis=1)).T if B is None else B
            
            r = np.random.random(self.sn)
            self.pi = r/r.sum() if pi is None else pi
    def update_para(self):
        
        if(not self._update_xi()):
            self.error = True
            return False
        if(not self._update_gamma()):
            self.error = True
            return False

        
        #update A    Todo
        sum_xi = self.xi.sum(axis=0)
        sum_gamma = self.gamma[:-1].sum(axis=0)
        A = (sum_xi.T/sum_gamma).T
        
        #update B   Todo
        sum_gamma = self.gamma.sum(axis=0)
        B = self.B.copy()
        for v in range(self.vn):
            idx = [i==v for i in self.learn_by]
            sum_gamma_v = self.gamma[idx].sum(axis=0)
            B[:,v] = sum_gamma_v/sum_gamma
        
        #update pi
        pi = self.gamma[0]
        
        if not np.sum(self.A-A) and not np.sum(self.B-B) and not np.sum(self.pi-pi):
            return False
        
        self.A = A
        self.B = B
        self.pi =pi
        
        #update α,β
        self.forward(self.learn_by)
        self.backward(self.learn_by)
        return True
    
    

    
    def learn_from_MOS(self,MOSes,max_step = 10,model='left_right',A=None,B=None,pi=None):
        
        self.model = model
        self.learn_by = MOSes
        
        self.A = A
        self.B = B
        self.pi = pi
        
        D = len(MOSes)
        
        is_init = True
        MOSes_FLATTEN = np.array([],dtype=self.dtype)
        for s in MOSes:
            MOSes_FLATTEN = np.concatenate((MOSes_FLATTEN,s))
        
        i=0
        while i<max_step:
            XI = np.zeros((1,self.sn,self.sn),dtype=self.dtype)
            gamma_A = np.zeros((1,self.sn),dtype=self.dtype)
            gamma_B = np.zeros((1,self.sn),dtype=self.dtype)
            gamma_PI = np.zeros((1,self.sn),dtype=self.dtype)
            
            for d in range(D):
                if is_init:
                    self._init_para(A,B,pi)
                
                self.learn_by = MOSes[d]
                self.forward(self.learn_by)
                self.backward(self.learn_by)
                
                self._update_xi()
                self._update_gamma()
#                print(XI,self.xi.copy())
                XI = np.concatenate((XI,self.xi.copy()),axis=0)
                gamma_A = np.concatenate((gamma_A,self.gamma[:-1].copy()),axis=0)
                gamma_B = np.concatenate((gamma_B,self.gamma.copy()),axis=0)
                gamma_PI = np.concatenate((gamma_PI,self.gamma[0][np.newaxis,:].copy()),axis=0)
            
            is_init = False
            
            gamma_A = gamma_A[1:]
            gamma_B = gamma_B[1:]
            gamma_PI = gamma_PI[1:]
            
            
            A = np.zeros_like(self.A)
            B = np.zeros_like(self.B)
            pi = np.zeros_like(self.pi)
            
            #update A    Todo
            SUM_XI = XI.sum(axis=0)
            SUM_gamma_A = gamma_A.sum(axis=0)
            A = (SUM_XI.T/SUM_gamma_A).T        
            
            #update B   Todo
            SUM_gamma_B = gamma_B.sum(axis=0)
            B = self.B.copy()
            for v in range(self.vn):
                idx = [i==v for i in MOSes_FLATTEN]
                sum_gamma_v = gamma_B[idx].sum(axis=0)
                B[:,v] = sum_gamma_v/SUM_gamma_B
            
            #update pi        
            pi = gamma_PI.sum(axis=0)/D
            
            self.A = A
            self.B = B
            self.pi = pi
            
            i+=1
        
        print('iter_time:',i)        


        
    def learn_para(self,observ_seq,max_step = 10,model='left_right',A=None,B=None,pi=None):
        
        self.model = model
        self.learn_by = observ_seq
        while self.error:
            self.error = False
            
            #随机初始化参数
            self._init_para(A,B,pi)
            
            self.forward(self.learn_by)
            self.backward(self.learn_by)
            
            i=1
            while(self.update_para() and i<max_step):
                i+=1
            
            if self.error:
                print("invalid value encountered, restart learning!")
            
        print('iter_time:',i)
    
    
    def evaluate(self,observ_seq,prob = True):
        self.forward(observ_seq)
        
        if not self.alpha[-1].sum():
            return -np.inf
        log_prob = -np.log2(self.c).sum()
        
        if prob:
            return 2**log_prob
        else:
            return log_prob
    
    
    def decoding(self,observ_seq):
        K = len(observ_seq)
        self.delta = np.zeros((K,self.sn),dtype=self.dtype) 
        
        self.delta[0] = self.pi*self.B[:,observ_seq[0]]
        State_s = np.zeros((K),dtype=np.int8)
        path = np.zeros((K,self.sn),dtype=np.int8) 
        for k in range(1,K):
            for i in range(self.sn):
                t = self.A[:,i]*self.B[i,observ_seq[k]]*self.delta[k-1]
                self.delta[k,i] = t.max()
                path[k,i]=t.argmax()
        State_s[-1] = self.delta[-1].argmax()
        
        for s in range(K-2,-1,-1):
            State_s[s] = path[s,State_s[s+1]]
        
        return State_s
    
    
    
    def show_info(self):
        print("name:%s\n"%self.name)

        print("A:\n",self.A.astype(np.float32))
        print("B:\n",self.B.astype(np.float32))
        print("pi:\n",self.pi.astype(np.float32))
        print("learn by:\n",self.learn_by)





def confusion_matrix(obvs,label,H,Prob = False):
    confusion_mat = np.zeros((len(H),len(H)),dtype=np.int8)
    
    for i in range(len(obvs)):
        ob = obvs[i]
        prob = [hmm.evaluate(ob,0) for hmm in H]
        clas = np.argmax(prob)
        confusion_mat[label[i],clas] += 1
        
        
    prob = (confusion_mat.T/confusion_mat.sum(axis=1)).T
    acc = prob[list(range(len(H))),list(range(len(H)))].sum()/len(H)
    if Prob:
        return prob,acc
    
    return confusion_mat,acc





def get_direction_feature(points,bin_num = 5,md=False):
    #输入字母，输出方向
#    degree_per_bin = 360//5
    points = np.array(points)[:,1:]
    vector = points[1:]-points[:-1]
    theta = np.arctan2(vector[:,1],vector[:,0])/np.pi*180 + 180
    obs = ((((theta)*bin_num)//360)%bin_num).astype(np.int8)
    
    if md:
        mod_obs = [obs[0]]
        for i in range(1,len(obs)):
            if obs[i] !=obs[i-1]:
                mod_obs.append(obs[i])
        mod_obs = np.array(mod_obs,dtype=np.int8)           
        return mod_obs
    else:
        return obs










def get_train_test1(bin_num):
    
    #输出拼接方向特征
    global df_Vowels
    c=0
    train_seq = []
    test_seq = []
    test_label = []
    for v in ['A','E','I','O','U']:
        t = []
        for i in range(20):
            t.extend(df_Vowels[v]['train'][i])
        cluster_seq = get_direction_feature(t,bin_num)
        train_seq.append(cluster_seq)
        
    
        for i in range(len(df_Vowels[v]['test'])):
            t = df_Vowels[v]['test'][i]
            cluster_seq =get_direction_feature(t,bin_num)
            test_seq.append(cluster_seq)    
            test_label.append(c)
        c+=1
    return train_seq,test_seq,test_label



def get_train_test2(bin_num):
    #输出Multi OS方向特征
    global df_Vowels
    c=0
    train_seq = []
    test_seq = []
    test_label = []

    for v in ['A','E','I','O','U']:
        for i in range(20):
            t = df_Vowels[v]['train'][i]
            cluster_seq = get_direction_feature(t,bin_num)
            train_seq.append(cluster_seq)
        
    
        for i in range(len(df_Vowels[v]['test'])):
            t = df_Vowels[v]['test'][i]
            cluster_seq =get_direction_feature(t,bin_num)
            test_seq.append(cluster_seq)    
            test_label.append(c)
        c+=1
    return train_seq,test_seq,test_label

def get_train_test3(K):
    

    #得到聚类坐标
    all_train_coord = np.array(get_train_coord(),dtype=np.int64)
    
    km = KMeans(all_train_coord)
    km.kmeans(K)
#    km.show_clusters()
    
    
    global df_Vowels
    c=0
    train_seq = []
    test_seq = []
    test_label = []
    
     
        
    for v in ['A','E','I','O','U']:
        for i in range(20):
            t = df_Vowels[v]['train'][i]
            cluster_seq = km.predict_seq(t)
            train_seq.append(cluster_seq)
        
    
        for i in range(len(df_Vowels[v]['test'])):
            t = df_Vowels[v]['test'][i]
            cluster_seq =km.predict_seq(t)
            test_seq.append(cluster_seq)    
            test_label.append(c)
        c+=1
    return train_seq,test_seq,test_label,km


def get_train_test4():
    #array序列
    
    global df_Vowels
    c=0
    train_seq = []
    test_seq = []
    test_label = []
    for v in ['A','E','I','O','U']:
        for i in range(20):
            t = df_Vowels[v]['train'][i]
            cluster_seq = np.array(t)[:,1:]
            cluster_seq = cluster_seq - cluster_seq.mean()
            train_seq.append(cluster_seq)    

        for i in range(len(df_Vowels[v]['test'])):
            t = df_Vowels[v]['test'][i]
            cluster_seq = np.array(t)[:,1:]
            cluster_seq = cluster_seq - cluster_seq.mean()
            test_seq.append(cluster_seq)    
            test_label.append(c)
        c+=1
    return train_seq,test_seq,test_label
    
with open(r'./data_pickle/df_Vowels_ori.pkl','rb') as f:
    df_Vowels = pickle.load(f)
# =============================================================================
# #聚类展示
# K = 5
# all_train_coord = np.array(get_train_coord(),dtype=np.int64)
# 
# km = KMeans(all_train_coord)
# km.kmeans(K)
# km.show_clusters()
# =============================================================================





# =============================================================================
# ##测试数据
# 
# seq = np.array([[0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
#                 [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]] )
# 
# A = np.array([
#     [0.1,  0.2,0.5, 0.2],
#     [.4, 0.1, .4, 0.1],
#     [0, .4, 0.4, .2],
#     [0.1, 0, .4, .5]])
# B = np.array([
#     [.5, .5],
#     [.3, .7],
#     [.6, .4],
#     [.8, .2]])
# pi = np.array([.25, .25, .25, .25])
# 
# h = HMM(4,2,name = 'A',A=A,B=B,pi=pi)
# 
# #h.learn_from_MOS(seq,max_step = 1,model=None)
# h.evaluate([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],False)
# h.decoding(seq[0])
# =============================================================================



# =============================================================================
# #拼接特征
# with open(r'./data_pickle/df_Vowels_ori.pkl','rb') as f:
#     df_Vowels = pickle.load(f)
#     
# N = 4
# K = 6
# train_seq,test_seq,test_label = get_train_test1(K)
# H=[]
# for i in range(5):
# #    bin_n = bin_num[i]
#     H.append(HMM(N,K,name = ['A','E','I','O','U'][i]))
#     
# for i in range(5):
#     H[i].learn_para(train_seq[i],2,None)
# 
# 
# accs = []
# for _ in range(5):
#     confusion_mat,acc = confusion_matrix(test_seq,test_label,H,1)
#     accs.append(acc)
# 
# print(np.average(accs))
# =============================================================================




# =============================================================================
# #Kmeans Multi特征
# with open(r'./data_pickle/df_Vowels_ori.pkl','rb') as f:
#     df_Vowels = pickle.load(f)
#     
# N = 4
# K = 8
# train_seq,test_seq,test_label,km = get_train_test3(K)
# H=[]
# for i in range(5):
# #    bin_n = bin_num[i]
#     H.append(HMM(N,K,name = ['A','E','I','O','U'][i]))
#     
# for i in range(5):
#     H[i].learn_from_MOS(train_seq[i*20:(i+1)*20],1,None)
# 
# 
# accs = []
# for _ in range(5):
#     confusion_mat,acc = confusion_matrix(test_seq,test_label,H,1)
#     accs.append(acc)
# 
# print(np.average(accs))
# =============================================================================




#方向 Multi特征
with open(r'./data_pickle/df_Vowels_ori.pkl','rb') as f:
    df_Vowels = pickle.load(f)
    
N = 4
K = 4
train_seq,test_seq,test_label = get_train_test2(K)
H=[]
for i in range(5):
#    bin_n = bin_num[i]
    H.append(HMM(N,K,name = ['A','E','I','O','U'][i]))
    
for i in range(5):
    H[i].learn_from_MOS(train_seq[i*20:(i+1)*20],1,None)


accs = []
for _ in range(2):
    confusion_mat,acc = confusion_matrix(test_seq,test_label,H,1)
    accs.append(acc)

print(np.average(accs))



# =============================================================================
# #进行实验，不同参数下的准确率
# ttt = list(range(2,8))
# 
# l_accs = []
# bin_num = 10
# N = 5
# for N in ttt:
#     ave_acc = 0
#     for _ in range(2):
#         train_seq,test_seq,test_label,km = get_train_test3(bin_num)
#         
#         A = HMM(N,bin_num,name = 'A')
#         E = HMM(N,bin_num,name = 'E')
#         I = HMM(N,bin_num,name = 'I')
#         O = HMM(N,bin_num,name = 'O')
#         U = HMM(N,bin_num,name = 'U')
#         H = [A,E,I,O,U]
#         
#         for i in range(5):
#             H[i].learn_from_MOS(train_seq[20*i:20*i+20],1,None)
#         
#         confusion_mat,acc = confusion_matrix(test_seq,test_label,H,1)
#         ave_acc += acc
#     l_accs.append(ave_acc/2)
# 
# 
# max_indx=np.argmax(l_accs)#max value index
# min_indx=np.argmin(l_accs)#min value index
# plt.plot(max_indx+2,l_accs[max_indx],'ks')
# show_max='['+str(100*np.float32(l_accs[max_indx]))+'%]'
# plt.annotate(show_max,xytext=(max_indx+2,l_accs[max_indx]),xy=(max_indx+2,l_accs[max_indx]))
# 
# plt.xlabel("Number of States")
# plt.ylabel("accuracy")
# plt.plot(ttt,l_accs)
# =============================================================================





# =============================================================================
# # DTW
# distance = lambda X,Y:np.sum((X-Y)**2)**0.5
# dtw = D.DTW()
# train_seq,test_seq,test_label = get_train_test4()
# dista = np.zeros((100,5))
# for i in range(len(test_seq)):
#     for j in range(5):
#         dis = 0
#         for k in range(20*j,20*j+20):
#             print(i,j,k)
#             dis += dtw.calc_distance(train_seq[k],test_seq[i])
#         dista[i,j] = dis
# 
# 
# confusion_mat = np.zeros((5,5))
# predict = dista.argmin(axis=1)
# 
# for i in range(5):
#     for j in range(i*20,i*20+20):
#         confusion_mat[test_label[j],predict[j]] += 1
#         
# prob = (confusion_mat.T/confusion_mat.sum(axis=1)).T
# acc = prob[list(range(len(prob))),list(range(len(prob)))].sum()/len(prob)
#         
# =============================================================================



# =============================================================================
# #decoding
# with open(r'./data_pickle/df_Vowels_ori.pkl','rb') as f:
#     df_Vowels = pickle.load(f)
#     
# N = 5
# dir_num = 4
# train_seq,test_seq,test_label = get_train_test2(dir_num)
# 
# hmm = HMM(N,dir_num,name = 'Super HMM')
# hmm.learn_from_MOS(train_seq,2,None)
# 
# decoding_train = []
# decoding_test = []
# for i in range(100):
#     decoding_train.append(hmm.decoding(test_seq[i]))
# 
# for i in range(100):
#     decoding_test.append(hmm.decoding(test_seq[i]))
# =============================================================================




##训练数据
#hmm.decoding(test_seq[0])
#    
#
#
#accs = []
#for _ in range(5):
#    confusion_mat,acc = confusion_matrix(test_seq,test_label,H,1)
#    accs.append(acc)
#
#print(np.average(accs))





