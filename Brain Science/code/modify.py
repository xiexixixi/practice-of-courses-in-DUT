# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:18:14 2019

    
@author: Lenovo
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import interpolate



loader = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\new_data_DataFrame2.pkl'
with open(loader,'rb') as file:
    new_data = pickle.load(file)


def rolling_mean(seq_,windows):
    
    seq = pd.Series(seq_)
    result = seq.rolling(windows,min_periods=1).mean()
    return result




def fft_transform(sp,windows_ = 15,plot = False,interv = 30):
    
    global new_data
        
    if isinstance(sp,tuple):
        dif = sp[1]-sp[0]
        start = sp[0]*100
        end = sp[1]*100
        end = start +interv*100
        dif = interv*100
    
    else:
        start = sp*100
        end = start +interv*100
        
        
    feature = []
#    feature_mean=[]
#    feature_arg = []    
    for channel in new_data.columns:
        if channel in ['1','2','3','4','5','6','7']:
            
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
            
#            if channel in ['5','6','7']:
#                feature.append(new_data[channel].mean())

#            argmax = sli[1:].argmax()
#            feature_arg.append(argmax)
            
    return start/100,end/100,feature




file_name = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\data\SC4002EC-Hypnogram.txt'
data_name = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\data\SC4002E0-PSG_data.txt'
text=''
file = open(file_name)

for line in file:
    text += line
    
file.close()

text = text.replace('\x14',' ')
text=text.replace('\x15',' ')
#text = text+' Sleep stage 1'


#转化成列表

stages = text.split('+')

def classify(string):
    sec = string.split(' ')
    print(sec)
    
    return sec[4],(int(sec[0]),int(sec[0])+int(sec[1]))


dic_stages = {'W':[],'R':[],'1':[],'2':[],'3':[],'4':[]}

for stage in stages:
    print(stage)
    if classify(stage)[0] in ['W','R','1','2','3','4']:
        label = classify(stage)[0]
        period = classify(stage)[1]
    dic_stages[label].append(period)
    
    
    
#导入所有数据
#new_data = pd.DataFrame()
#with open(data_name,'rb') as file:
#    new_data=pd.read_csv(file,sep=',')





new_data['4'].interpolate(inplace = True)
new_data['4'] = new_data['4'] - new_data['4'].mean()
new_data['4'] = new_data['4']/(new_data['4'].max()-new_data['4'].min())





dic_extract_feature={'W':[],'R':[],'1':[],'2':[],'3':[],'4':[]}


for stage_label,stages_period in dic_stages.items():
    for sp in stages_period:
        p = (sp[1]-sp[0])/30
        sp_part = np.linspace(sp[0],sp[1],p+1)
        for i in range(len(sp_part)-1):
            result = fft_transform(sp_part[i],15)
            dic_extract_feature[stage_label].append(result)

#存文件
dumper = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\new_data_DataFrame2.pkl'
with open(dumper,'wb') as file:
    pickle.dump(new_data,file)
    
dumper = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\dic_stages_dict2.pkl'
with open(dumper,'wb') as file:
    pickle.dump(dic_stages,file)

dumper = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\dic_extract_feature_dict2.pkl'
with open(dumper,'wb') as file:
    pickle.dump(dic_extract_feature,file)


    
    
    
    
    
    
    
    
    
    
    
    
    
