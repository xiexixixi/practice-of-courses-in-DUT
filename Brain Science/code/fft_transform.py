# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:26:00 2019

@author: Lenovo
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

loader = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\new_data_DataFrame.pkl'
with open(loader,'rb') as file:
    new_data = pickle.load(file)

loader = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\dic_stages_dict.pkl'
with open(loader,'rb') as file:
    dic_stages = pickle.load(file)

loader = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\stages_list.pkl'
with open(loader,'rb') as file:
    stages = pickle.load(file)

loader = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\dic_extract_feature_dict.pkl'
with open(loader,'rb') as file:
    dic_extract_feature = pickle.load(file)

def fft_plot(p):
    t2=np.abs(np.fft.rfft(p,norm = 'ortho'))
    plt.plot(t2)
    return t2
    


def corr(seq):
    
    df = pd.DataFrame()
    for i in range(len(seq)):
        df[i]=seq[i]
    c = df.corr()
    print(c)
    return c


def rolling_mean(seq_,windows):
    
    seq = pd.Series(seq_)
    result = seq.rolling(windows,min_periods=1).mean()
    return result
    
def fft_transform_cor(sp,windows_ = 10,plot = False):
    
    global new_data
    
    dif = sp[1]-sp[0]
    start = sp[0]*100
    end = sp[1]*100
    print(start,end)

    feature_mean=[]
    feature_arg = []
    
    for channel in new_data.columns:
        if channel != 'Time':
#            print(channel)
            p = new_data.loc[start:end-1,channel]
#            print(len(p))
            tsfmd = np.abs(np.fft.rfft(p,norm = 'ortho'))
#            print(len(tsfmd))
            m_tsfmd = rolling_mean(tsfmd,windows = windows_)
            if plot and channel == '2':
#                plt.plot(tsfmd)
                plt.plot(m_tsfmd)
                
            feature_mean.append(p.mean())
            argmax = m_tsfmd[1:].argmax()
            feature_arg.append(argmax)
    feature_arg=(np.array(feature_arg)/dif).tolist()
#    feature_mean=(np.array(feature_mean)/dif).tolist()
    feature_mean.pop()
    feature_mean.pop()   
    return m_tsfmd


def fft_transform(sp,windows_ = 15,plot = False,interv = 30):
    
    global new_data
    

    dif = sp[1]-sp[0]
    start = sp[0]*100
    end = sp[1]*100
    
    end = start +interv*100
    dif = interv*100
    
    feature = []
#    feature_mean=[]
#    feature_arg = []    
    for channel in new_data.columns:
        if channel in ['1','2','3','4','567']:
            
#            print(channel)
            p = new_data.loc[start:end-1,channel]
#            print(len(p))
            tsfmd = np.abs(np.fft.rfft(p,norm = 'ortho'))
#            print(len(tsfmd))
            m_tsfmd = rolling_mean(tsfmd,windows = windows_)
            sli = rolling_mean(m_tsfmd,windows = 3)

            if plot and channel == '1':
                plt.plot(tsfmd)
                
                plt.plot(sli,color=(0.8,0.5,0.1),linewidth=5)
            
            if channel in ['1','2','3']:
                feature.append(sli.max())
                feature.append(sli[200:250].max())
                
            if channel == '4':
                feature.append(sli[10])

            argmax = sli[1:].argmax()
#            feature_arg.append(argmax)
            
    return sp[0],sp[1],feature




fea=[]
l=[]
for label,data in dic_extract_feature.items():
    for i in data:
        l=[]
        l.append(label)
        l.append(i[2])
        
        fea.append(l)


random.shuffle(fea)

train = fea[0:int(len(fea)*0.8)]

test = fea[int(len(fea)*0.8):]







'''

V = 'R'
w = 15
for i in range(1):
    fft_transform(dic_stages[V][i],w,plot=True)
'''

'''
dic_extract_feature={'W':[],'R':[],'1':[],'2':[],'3':[],'4':[]}

for stage_label,stages_period in dic_stages.items():
    for sp in stages_period:
        
        result = fft_transform(sp,15)
        dic_extract_feature[stage_label].append(result)
'''