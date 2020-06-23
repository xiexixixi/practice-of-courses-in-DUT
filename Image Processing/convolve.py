# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:47:48 2020

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pli



    
# mode=0 
def Convolve(I,F,iw,ih,fw,fh,Kernel_mode = 0):
    
    Conv_I = np.zeros_like(I)

    if Kernel_mode == 0:
        Image_pad = np.pad(I, ((0,fh-1),(fw-1,0) ), 'constant', constant_values=0)
        func = lambda x,y:np.sum(Image_pad[x:x+fh,y:y+fw]*F)
        for i in range(ih):
            for j in range(iw):
                Conv_I[i,j] = func(i,j)
                
                
    elif Kernel_mode == 1:
        Image_pad = np.pad(I, ((0,fh-1),(fw-1,0) ), 'reflect')
        func = lambda x,y:np.sum(Image_pad[x:x+fh,y:y+fw]*F)
        for i in range(ih):
            for j in range(iw):
                Conv_I[i,j] = func(i,j)
                

    elif Kernel_mode == 2:
        Image_pad = np.pad(I, ((0,fh-1),(fw-1,0) ), 'constant', constant_values=0)
        func = lambda x,y:np.sum(Image_pad[x:x+fh,y:y+fw]*F)

        for i in range(ih):
            for j in range(iw):
                
                bottom = np.clip(ih-i,0,3)
                left = np.clip(fw-j-1,0,3)
                
                s = np.sum(F[:bottom,left:])
                Conv_I[i,j] = int(np.round(func(i,j)/s))
                
    return Conv_I
                

if __name__ == '__main__':
    
    im1_path = r'C:\Users\Lenovo\Desktop\flowergray.jpg'
    im1 = pli.imread(im1_path)
    F = np.array([[0,-0.25,0],
                  [-0.25,1,-0.25],
                  [0,-0.25,0]])
    FF = np.array([[1/4]*4]).reshape(2,2)
       
    FFF = np.array([[1/9]*9]).reshape(3,3)
    
    ih,iw = im1.shape
    
    fh,fw = FFF.shape
    O = Convolve(im1,FFF,iw,ih,fw,fh,Kernel_mode = 1)
    
    OO = Convolve(im1,FFF,iw,ih,fw,fh,Kernel_mode = 1)
    OOO = Convolve(im1,FFF,iw,ih,fw,fh,Kernel_mode = 2)
    
    
    
    plt.subplot(1,3,1) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
    plt.imshow(im1/255,cmap='gray')
    plt.title('origin picture')
    plt.subplot(1,3,2) #两行两列,这是第二个图
    plt.imshow(O,cmap='gray')
    plt.title('2x2 filtered')
    
    plt.subplot(1,3,3)#两行两列,这是第三个图
    plt.imshow(OO,cmap='gray')
    plt.title('3x3 filtered')
    
    
    #
    #plt.subplot(1,3,1) 
    #plt.imshow(O,cmap='gray')
    #plt.title('method 1')
    #plt.subplot(1,3,2) 
    #plt.imshow(OO,cmap='gray')
    #plt.title('method 2')
    #
    #plt.subplot(1,3,3)#两行两列,这是第三个图
    #plt.imshow(OOO,cmap='gray')
    #plt.title('method 3')






