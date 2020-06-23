# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:24:38 2020

@author: Lenovo
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pli
from PIL import Image

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
    
    if original_image.ndim==2:
        derive = np.zeros((int(target_size[0]),int(target_size[1])),dtype = int)
        for i in range(derive.shape[0]):
            for j in range(derive.shape[1]):
                derive[i,j] = interplotation((i,j),derive.shape,original_image)    
    else:
        derive = np.zeros((int(target_size[0]),int(target_size[1]),3),dtype = int)
        R,G,B = original_image[:,:,0],original_image[:,:,1],original_image[:,:,2]
        for i in range(derive.shape[0]):
            for j in range(derive.shape[1]):
                derive[i,j,0] = interplotation((i,j),derive.shape,R)
                derive[i,j,1] = interplotation((i,j),derive.shape,G)
                derive[i,j,2] = interplotation((i,j),derive.shape,B)

    return derive

def imresize2(original_image,d,target_size):

    i = Image.fromarray(original_image)
    i = i.resize(target_size[::-1],Image.BILINEAR)
    image = np.asarray(i)
    return image


def RGB2Gray(Image):
    row,col = Image.shape[:2]
    print(row,col)
    gray = np.zeros((row,col),dtype = 'float')
    gray = 0.11 *Image[:,:,0].squeeze() +0.59*Image[:,:,1].squeeze() +0.3*Image[:,:,2].squeeze() 
    return gray.astype('uint8')


def Convolve(I,F,iw,ih,fw,fh,Kernel_mode = 1,down_sampling_step = 1):
    
    Conv_I = np.zeros_like(I)

    if Kernel_mode == 0:
        Image_pad = np.pad(I, ((0,fh-1),(fw-1,0) ), 'constant', constant_values=0)
        func = lambda x,y:np.sum(Image_pad[x:x+fh,y:y+fw]*F)
        for i in range(0,ih,down_sampling_step):
            for j in range(0,iw,down_sampling_step):
                Conv_I[i,j] = func(i,j)
                
                
    elif Kernel_mode == 1:
        Image_pad = np.pad(I, ((0,fh-1),(fw-1,0) ), 'reflect')
        func = lambda x,y:np.sum(Image_pad[x:x+fh,y:y+fw]*F)
        for i in range(0,ih,down_sampling_step):
            for j in range(0,iw,down_sampling_step):
                Conv_I[i,j] = func(i,j)
                

    elif Kernel_mode == 2:
        Image_pad = np.pad(I, ((0,fh-1),(fw-1,0) ), 'constant', constant_values=0)
        func = lambda x,y:np.sum(Image_pad[x:x+fh,y:y+fw]*F)

        for i in range(0,ih,down_sampling_step):
            for j in range(0,iw,down_sampling_step):
                
                bottom = np.clip(ih-i,0,3)
                left = np.clip(fw-j-1,0,3)
                
                s = np.sum(F[:bottom,left:])
                Conv_I[i,j] = int(np.round(func(i,j)/s))
                
    return Conv_I
                

def Convolve_RGB(I,F,iw,ih,fw,fh,Kernel_mode = 1,down_sampling_step = 1):

    R,G,B = I[:,:,0],I[:,:,1],I[:,:,2]
    C_R = Convolve(R,F,iw,ih,fw,fh,Kernel_mode,down_sampling_step)
    C_G = Convolve(G,F,iw,ih,fw,fh,Kernel_mode,down_sampling_step)
    C_B = Convolve(B,F,iw,ih,fw,fh,Kernel_mode,down_sampling_step)
    
    Conv_I = np.concatenate((C_R[:,:,np.newaxis],C_G[:,:,np.newaxis],C_B[:,:,np.newaxis]),axis = 2)
    return Conv_I


def down_sampling(Image,step = 2):
    return Image[::step,::step]
    
    
def Guassianian_floor(Image,F):
    
    if Image.ndim == 2:
    
        ih,iw = Image.shape
        fh,fw = F.shape
        C = Convolve(Image,F,iw,ih,fw,fh,Kernel_mode = 1)
        
        next_floor = down_sampling(C,step = 2)
        return next_floor
    
    else:
        ih,iw = Image.shape[0],Image.shape[1]
        fh,fw = F.shape
        C = Convolve_RGB(Image,F,iw,ih,fw,fh,Kernel_mode = 1)
        
        next_floor = down_sampling(C,step = 2)
        return next_floor

def Guassianian_pyramid(Image,F,floor_num = 20):
    
    print('Constructing Guassianian pyramid...')
    print('Input :',Image.shape)
    
    I = Image.copy()
    G_pyramid = [I]
    
    if floor_num == 1:
        print('only 1 floor?')
        return G_pyramid
    
    floor_num -= 2
    
    I = Guassianian_floor(I,F).copy()
    while I.shape[0] != 1 and I.shape[1]!=1 and floor_num:
        print("constructing ",I.shape)

        G_pyramid.append(I)
        I = Guassianian_floor(I,F).copy()
        
        
        
        floor_num -=1
    
    G_pyramid.append(I)
    
    print('constructions complete')
    return G_pyramid
    

def Laplacian_pyramid(G_pyramid,F,start=0):
    
    print('Constructing Laplacian pyramid...')

    
    L_pyramid =[]
    
    for i in range(start,len(G_pyramid)-1):
        flagr,flagc =0,0
        r = 1-G_pyramid[i].shape[0]%2
        c = 1-G_pyramid[i].shape[1]%2        
        if G_pyramid[i+1].shape[0] == 1 or G_pyramid[i+1].shape[1] == 1:
            if G_pyramid[i+1].shape[0] == 1:
                flagr = 1
                I = np.concatenate((G_pyramid[i+1],G_pyramid[i+1]),axis=0)
                
            if G_pyramid[i+1].shape[1] == 1:
                flagc = 1
                I = np.concatenate((G_pyramid[i+1],G_pyramid[i+1]),axis=0)
            
            if flagr and not flagc:
                I = imresize(I,I.shape,(2,G_pyramid[i].shape[1]-c))
            elif flagc and not flagr:
                I = imresize(I,I.shape,(G_pyramid[i].shape[0]-r,2))

            
        #判定奇偶，确定是否填充
        
        else:
            r = 1-G_pyramid[i].shape[0]%2
            c = 1-G_pyramid[i].shape[1]%2
            I = imresize(G_pyramid[i+1],G_pyramid[i+1].shape,(G_pyramid[i].shape[0]-r,G_pyramid[i].shape[1]-c))
            if I.ndim==2:
                I = np.pad(I, ((0,r),(0,c)), 'reflect')
            elif I.ndim ==3:
                print(1)
                I = np.pad(I, ((0,r),(0,c),(0,0)), 'reflect')
                print(I.shape)

        print("constructing ",I.shape)

        ih,iw = I.shape[0],I.shape[1]
        fh,fw = F.shape
        if I.ndim==2:
            I = Convolve(I,F,iw,ih,fw,fh,Kernel_mode = 1)
            
            L_pyramid.append(G_pyramid[i]-I)

        if I.ndim==3:
            I = Convolve_RGB(I,F,iw,ih,fw,fh,Kernel_mode = 1)
            print(I.max(),I.min(),I.dtype)
            print(G_pyramid[i].max(),G_pyramid[i].min(),G_pyramid[i].dtype)
            l = G_pyramid[i]-I
                
            L_pyramid.append(l)
    
    print('construction complete')
    return L_pyramid
    


def change(I,a,d=1):
    I=I-(I==1)
    I = I+a
    I = I.astype('float32')
    I = I/I.max(axis=2)[:,:,np.newaxis]
    I = I.astype('float32')/d
    
    plt.imshow(I)
    return I


def scale(r):
    r = r.copy()
    r = r- r.min()
    r = r/r.max()
    return r
def scale_image(I):
    r = scale(I[:,:,0])
    g = scale(I[:,:,1])
    b = scale(I[:,:,2])

    i = np.concatenate((r[:,:,np.newaxis],g[:,:,np.newaxis],b[:,:,np.newaxis]),axis=2)
    return i




def show(I,gray = 0):
    I = I.copy()
    if I.ndim==2:
        plt.imshow(I,cmap='gray')
    elif gray:

        I = RGB2Gray(I)
        plt.imshow(I,cmap = 'gray')
    else:
        I = scale_image(I)
        plt.imshow(I)

        

def rebuild(G,L,F,i):
    r = 1-G[i].shape[0]%2
    c = 1-G[i].shape[1]%2    
    I = imresize(G[i+1],G[i+1].shape,(G[i].shape[0]-r,G[i].shape[1]-c))


    if I.ndim==2:
        I = np.pad(I, ((0,r),(0,c)), 'reflect')
    elif I.ndim ==3:
        I = np.pad(I, ((0,r),(0,c),(0,0)), 'reflect')

    ih,iw = I.shape[0],I.shape[1]
    fh,fw = F.shape
    if I.ndim==2:
        I = Convolve(I,F,iw,ih,fw,fh,Kernel_mode = 1)
        
        image = I+L[i]
        return image

    if I.ndim==3:
        I = Convolve_RGB(I,F,iw,ih,fw,fh,Kernel_mode = 1)
        image = I+L[i]
        return image
    
def pictures(i,j):
    
    for k in range(1,8):
        plt.axis('off')
        plt.subplot(2,7,k) #这是第一个图plt.subplot('行','列','编号')
        plt.imshow(i[k],cmap='gray')
        
    for k in range(8,15):
        plt.axis('off')
        plt.subplot(2,7,k) #这是第一个图plt.subplot('行','列','编号')
        
#        I = scale_image(j[k-7])
        plt.imshow(j[k-7],cmap='gray')
        plt.axis('off')



if __name__ == '__main__':
    im1_path = r'C:\Users\Lenovo\Desktop\building.jpg'
#    im1_path = r'C:\Users\Lenovo\Desktop\B408156B90EDFC5DBCCFE80B6EB98DA1.jpg'
#    im1_path = r'C:\Users\Lenovo\Desktop\crooked_horizon.jpg'
#    im1_path = r'C:\Users\Lenovo\Desktop\B48CB2953A6308FDD5C86573F7A73C1A.jpg'
#    im1_path = r'C:\Users\Lenovo\Desktop\flowergray.jpg'
    
    im1 = pli.imread(im1_path)
    im1 = RGB2Gray(im1)
#    im1 = imresize2(im1,0,(512,512))
    print(im1.shape)


    F = np.array([[1/9]*9]).reshape(3,3)
    i = Guassianian_pyramid(im1,F)
    
    j = Laplacian_pyramid(i,F,0)

    re = rebuild(i,j,F,0)
    pictures(i,j)