# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:50:00 2019

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import feature_extraction1 as fe
 
'''
ax = plt.figure().add_subplot(111, projection = '3d')


xs = fe.all_data['5']
ys = fe.all_data['6']
zs = fe.all_data['7']


#ax.scatter(xs, ys, zs, c = 'r', marker = 'o') #点为红色三角形



#设置坐标轴

ax.set_xlabel('EMG submental')

ax.set_ylabel('temp rectal')

ax.set_zlabel('event marker')
#显示图像
'''

'''
fig=plt.figure()
ax1=fig.add_subplot(221) #2*2的图形 在第一个位置


ax1.set_title('EEG Fpz-Cz hist')

ax1.hist(fe.all_data['1'],bins=1000,color='b')



ax2=fig.add_subplot(222)

ax2.set_title('EEG Pz-Oz hist')

ax2.hist(fe.all_data['2'],bins=1000,color = 'green')




ax3=fig.add_subplot(223)

ax3.set_title('EOG horizontal hist')

ax3.hist(fe.all_data['3'],bins=1000,color = 'orange')




ax4=fig.add_subplot(224)

ax4.set_title('Resp oro-nasal')

ax4.hist(fe.all_data['4'],bins=1000,color = 'red')

plt.show()

'''


fig=plt.figure()
ax1=fig.add_subplot(221) #2*2的图形 在第一个位置


ax1.set_title('EMG submental hist')

ax1.hist(fe.all_data['5'],bins=1000,color='b')



ax2=fig.add_subplot(222)

ax2.set_title('Temp rectal hist')

ax2.hist(fe.all_data['6'],bins=1000,color = 'green')




ax3=fig.add_subplot(223)

ax3.set_title('Event marker hist')

ax3.hist(fe.all_data['7'],bins=1000,color = 'orange')



ax4=fig.add_subplot(224)

ax4.set_title('After PCA process')

ax4.hist(fe.new_data['567'],bins=1000,color = 'red')




plt.show()



