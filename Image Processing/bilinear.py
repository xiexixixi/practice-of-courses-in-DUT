# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:28:49 2020

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg

im1_path = r'C:\Users\Lenovo\Desktop\flowergray.jpg'

im1 = plimg.imread(im1_path)
print(im1.shape)
#%matplotlib qt5
plt.imshow(im1,cmap='gray')


def interplotation(coord,derive_size,origin_img):

    delta_x = (origin_img.shape[0]-1)/(derive_size[0]-1)
    delta_y = (origin_img.shape[1]-1)/(derive_size[1]-1)
    x = coord[0]*delta_x
    y = coord[1]*delta_y
    
    gray1 = 0
    gray2 = 0
    gray_level = 0
    
    x1 = int(np.floor(x))
    x2 = int(np.ceil(x))
    y1 = int(np.floor(y))
    y2 = int(np.ceil(y)) 
        
    if(x1==x2):
        gray1 = origin_img[x1][y1]
        gray2 = origin_img[x1][y2]
    else:
        gray1 = (x-x1)*origin_img[x1][y2]+(x2-x)*origin_img[x1][y1]
        gray2 = (x-x1)*origin_img[x2][y2]+(x2-x)*origin_img[x2][y1]
    if(y1==y2):
        gray_level = gray1
    else:
        gray_level = (y-y1)*gray2+(y2-y)*gray1
    
    return gray_level
    


def bilinear(k,origin):
    derive = np.zeros((int(k*origin.shape[0]),int(k*origin.shape[1])),dtype = int)
    for i in range(derive.shape[0]):
        for j in range(derive.shape[1]):
            derive[i,j] = interplotation((i,j),derive.shape,origin)
    return derive

def imresize(original_image,original_size,target_size):
    
    derive = np.zeros((int(target_size[0]),int(target_size[1])),dtype = int)
    for i in range(derive.shape[0]):
        for j in range(derive.shape[1]):
            derive[i,j] = interplotation((i,j),derive.shape,original_image)    
    
    return derive

k = 1.5

#%matplotlib qt5

plt.imshow(imresize(im1,im1.shape,[int(k*i) for i in im1.shape]),cmap = 'gray')

k=0.75
plt.imshow(imresize(im1,im1.shape,[int(k*i) for i in im1.shape]),cmap = 'gray')


