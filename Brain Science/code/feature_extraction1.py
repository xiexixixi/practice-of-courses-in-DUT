# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:53:02 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import interpolate

file_name = r'C:\Users\Lenovo\Desktop\SC4001EC-Hypnogram .txt'

text='0 30630 '
file = open(file_name)

for line in file:
    text += line
    
file.close()

text = text.replace('\x14',' ')
text=text.replace('\x15',' ')
#text = text.replace(', +',' ')
text = text+' Sleep stage 1'


#转化成列表

stages = text.split('+')

def classify(string):
    sec = string.split(' ')
    print(sec)
    return sec[4],(int(sec[0]),int(sec[0])+int(sec[1]))

dic_stages = {'W':[],'R':[],'1':[],'2':[],'3':[],'4':[]}

for stage in stages:
    label = classify(stage)[0]
    period = classify(stage)[1]
    dic_stages[label].append(period)


#读取数据内容
all_data_PKL = r'C:\Users\Lenovo\Desktop\大二CS\大二小学期 复变+汇编+程序设计\脑科学\代码\all_data_DataFrame.pkl'
with open(all_data_PKL,'rb') as file:
    all_data = pickle.load(file)
    
    
#EDA
#后三列相关性大，PCA进行降维
#PCA
#if __name__ == '__main__':
    
data = all_data.loc[:,'5':'7']
data.dropna(inplace=True)
data_scaled = data-data.mean(axis=0)

data_scaled[data_scaled['7']<-200]=np.nan
data_scaled.fillna(0,axis=1,inplace=True)

diff = data_scaled.max(axis=0)-data_scaled.min(axis=0)

data_scaled = data_scaled/diff


pca = PCA(1)
pca.fit(data_scaled)

low_d = pca.transform(data_scaled) #降低维度

#将前三列归一化
#data =  all_data.loc[:,'1':'4']
#data = data - data.mean(axis = 0)

#diff = data.max(axis=0)-data.min(axis=0)

#data_scaled = data/diff
x = np.arange(0,79500)
f = interpolate.interp1d(x, np.squeeze(low_d))
xnew = np.arange(0, 79499, 0.01)

low_d_new=f(xnew)




new_data = all_data.loc[:7949900-1,'Time':'4']

new_data['567']=low_d_new
new_data['4'].interpolate(inplace = True)

new_data['4'] = new_data['4'] - new_data['4'].mean()


new_data['4'] = new_data['4']/(new_data['4'].max()-new_data['4'].min())

#后四列归一化完成，下面进行存储






