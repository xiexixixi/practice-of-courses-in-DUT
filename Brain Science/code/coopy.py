# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:06:51 2019

@author: Lenovo
"""

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



loader = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\all_data_DataFrame.pkl'
with open(loader,'rb') as file:
    new_data = pickle.load(file)
    
    
loader = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\dic_stages_dict.pkl'
with open(loader,'rb') as file:
    dic_stages = pickle.load(file)

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
dumper = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\new_data_DataFrame.pkl'
with open(dumper,'wb') as file:
    pickle.dump(new_data,file)
    
dumper = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\dic_stages_dict.pkl'
with open(dumper,'wb') as file:
    pickle.dump(dic_stages,file)

dumper = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\dic_extract_feature_dict.pkl'
with open(dumper,'wb') as file:
    pickle.dump(dic_extract_feature,file)


    
    
    
    
    
    
    
    
    
    
    
    
    
