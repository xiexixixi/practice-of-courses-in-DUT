# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:12:02 2020

@author: Lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pli
import pickle
from PIL import Image
import os

def Convolve(I,F):
    iw,ih = I.shape
    fw,fh = F.shape
    Conv_I = np.zeros_like(I)
    Image_pad = np.pad(I,(((fh-1)//2,(fh-1)//2),((fw-1)//2,(fw-1)//2)), 'reflect')
    func = lambda x,y:np.sum(Image_pad[x:x+fh,y:y+fw]*F)
    for i in range(ih):
        for j in range(iw):
            Conv_I[i,j] = func(i,j)
    return Conv_I

def downsample(I,step = 2):
    return I[::step,::step]

def RGB2Gray(image):
    row,col = image.shape[:2]
    print(row,col)
    gray = np.zeros((row,col),dtype = 'float')
    gray = 0.11 *image[:,:,0].squeeze() +0.59*image[:,:,1].squeeze() +0.3*image[:,:,2].squeeze() 
    print(gray.shape,1)
    return gray.astype('uint8')

def imresize(original_image,target_size):

    i = Image.fromarray(original_image)
    ii = i.resize(target_size[::-1],Image.BILINEAR)
    image = np.asarray(ii)
    return image

def show(I,gray = 0):
    I = I.copy()
    if I.ndim==2:
        plt.imshow(I,cmap='gray')
    elif gray:
        I = RGB2Gray(I)
        plt.imshow(I,cmap = 'gray')
    else:
        plt.imshow(I)
        
def display(GP,optave_num=3,scale_num=6):
    
    counter = 1
    for i in range(optave_num):
        for j in range(scale_num):
            plt.subplot(optave_num,scale_num,counter) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
            plt.imshow(GP[i][j],cmap='gray')
            counter+=1

def Make_pickle(img_path,name1,name2):
    
    K = Guassian_Kernel(1.60,5)
    
    img =  pli.imread(img_path)
    print(img.ndim)
    if img.ndim==3:
        print('convert to gray')
        img = RGB2Gray(img)
    
    GP,DOG = generate_DOG(img,6,sigma0=1.5)
    
    path = r'C:\Users\Lenovo\Desktop\pickle\\'
    with open(path+name1+'.pkl','wb') as f:
        pickle.dump(GP,f)
    print('GP finish')
    with open(path+name2+'.pkl','wb') as f:
        pickle.dump(GP,f)
    print('DOG finish')
            

def load_pickle(GP_file,DOG_file):
    
    path = r'C:\Users\Lenovo\Desktop\pickle\\'
    with open(path+GP_file+'.pkl','wb') as f:
        GP =pickle.load(f)
    with open(path+DOG_file+'.pkl','wb') as f:
        DOG =pickle.load(f)
    return GP,DOG



        
def Guassian_Kernel(sigma,dim):

    temp = [t - (dim//2) for t in range(dim)]
    assistant = []
    for i in range(dim):
        assistant.append(temp)
    assistant = np.array(assistant)
    temp = 2*sigma*sigma
    kernel = (1.0/(temp*np.pi))*np.exp(-(assistant**2+(assistant.T)**2)/temp)
    return kernel

#生成一层的高斯图像
#sigma0是该octave的第一层sigma
def GP_Octave(I,Scale_num,k,sigma0):
    octave = []
#     sigma = [k**i for i in range(Scale_num)]*sigma0
#     kernel_dim = [int(6*i+2) if (6*i)%2 else int(6*i+1) for i in sigma]
    m = min(I.shape)
    for i in range(Scale_num):
        sigma = k**i * sigma0
        kernel_dim = int(6*sigma+2) if int(6*sigma)%2 else int(6*sigma+1)
        if kernel_dim > m:
            kernel_dim = int(m+2) if int(m)%2 else int(m)+1
        print(i,kernel_dim)

        K = Guassian_Kernel(sigma,kernel_dim)
        octave.append(Convolve(I,K))
    
    return octave
    

def generate_DOG(I,Scale_num,sigma0=1.5,Octave_num=None):
    
    if(I.ndim==3):
        Image = RGB2Gray(I)
    else:
        Image = I.copy()

    n = Scale_num-3
    if Octave_num is None:
        Octave_num = int(np.log2(min(Image.shape[0],Image.shape[1]))) - 3
    
    k = 2**(1./n)
    G_Pyramid = []
    init_sigma = [sigma0*(2**i) for i in range(Octave_num)]
#     kernel_dim = [int(6*i+2) if (6*i)%2 else int(6*i+1) for i in l]
    for i in range(Octave_num):
        G_Pyramid.append(GP_Octave(Image,Scale_num,k,init_sigma[i]))
        Image = downsample(Image,step = 2)
        
    DOG = [[G_Pyramid[i][j+1].astype(int) - G_Pyramid[i][j].astype(int) for j in range(len(G_Pyramid[0])-1)] for i in range(len(G_Pyramid))]
    return G_Pyramid,DOG


#第二步，迭代找到极小点精确位置

def Extreme_points(DOG):

    sigma = 1.6
    Key_Points=[]
    O = len(DOG)
    S = len(DOG[0])
    n = S - 3
    for o in range(O):
        for s in range(1,S-1):
            threshold =255*0.5*0.04/n
            I_pre, I_cur,I_next = DOG[o][s-1],DOG[o][s],DOG[o][s+1]
            stride = max(1,2**(int(np.log2(min(I_cur.shape)))-6))
            print(stride)
            for i in range(1,I_cur.shape[0]-1,stride):
                for j in range(1,I_cur.shape[1]-1):
                    val = I_cur[i,j]
                    eight_neiborhood_prev = I_pre[i-1:i+2,j-1:j+2]
                    eight_neiborhood = I_cur[i-1:i+2,j-1:j+2]
                    eight_neiborhood_next = I_next[i-1:i+2,j-1:j+2]
                    if np.abs(val) > threshold and \
                        ((val > 0 and (val >= eight_neiborhood_prev).all() and (val >= eight_neiborhood).all() and (val >= eight_neiborhood_next).all())
                         or (val < 0 and (val <= eight_neiborhood_prev).all() and (val <= eight_neiborhood).all() and (val <= eight_neiborhood_next).all())):
#                    if gray_val> threshold\
#                    and ((gray_val>0 and (gray_val>=pixels_pre).all() and (gray_val>=pixels_cur).all() and (gray_val>=pixels_next).all())\
#                    or (gray_val < 0 and (gray_val<=pixels_pre).all() and (gray_val<=pixels_cur).all() and (gray_val<=pixels_next).all())):
                        K_point,x,y,scale = adjustLocalExtrema(DOG,o,s,i,j)
                    
                        if K_point is None:
                            continue
#                        if K_point != None:
#                            print(x,y,scale)
#                            return [K_point]
                        Key_Points.append(K_point)
                        
    
    return Key_Points



def adjustLocalExtrema(DOG,o,s,x,y):
    iter_steps = 5
    img_border = 5
    n=3
    I = DOG[o][s]
    
    for i in range(iter_steps):
        #迭代点边界判断
        if s < 1 or s > n or y < img_border or y >= I.shape[1] - img_border or x < img_border or x >= I.shape[0] - img_border:
            return [None]*4
        I_prev = DOG[o][s-1].copy()
        I = DOG[o][s].copy()
        I_next = DOG[o][s+1].copy()
        
        dD = np.array([I[x,y + 1] - I[x, y - 1] ,
                       I[x + 1, y] - I[x - 1, y],
                       I_next[x, y] - I_prev[x, y]],dtype=float)*0.5
            
        
        v2 = I[x, y] * 2
        Dxx = (I[x, y + 1] + I[x, y - 1] - v2).astype(float)
        Dyy = (I[x + 1, y] + I[x - 1, y] - v2).astype(float)
        Dss = (I_next[x, y] + I_prev[x, y] - v2).astype(float)
        Dxy = (I[x + 1, y + 1] - I[x + 1, y - 1] - I[x - 1, y + 1] + I[x - 1, y - 1]).astype(float) * 0.25
        Dxs = (I_next[x, y + 1] - I_next[x, y - 1] - I_prev[x, y + 1] + I_prev[x, y - 1]).astype(float) * 0.25
        Dys = (I_next[x + 1, y] - I_next[x - 1, y] - I_prev[x + 1, y] + I_prev[x - 1, y]).astype(float) * 0.25
        H=np.array([[Dxx, Dxy, Dxs],
                    [Dxy, Dyy, Dys],
                    [Dxs, Dys, Dss]],dtype=float)
    
        
        X = -np.matmul(np.linalg.pinv(H),dD)
        dc,dr,ds = X

        if np.abs(ds) < 0.5 and np.abs(dr) < 0.5 and np.abs(dc) < 0.5:
            break
        if (np.abs(X) < 0.5).all():
            break


        y += int(np.round(dc))
        x += int(np.round(dr))
        s += int(np.round(ds))
    
    #迭代 5 次都没找到，丢弃
    else:
        return [None]*4
    
    
    #判断找到的点是否在边界内，边界外舍去
    if s < 1 or s > n or y < img_border or y >= I.shape[1] - img_border or x < img_border or x >= \
            I.shape[0] - img_border:
        return [None]*4

    #判断是可能是噪声，可能是噪声舍去
    dg = dD.dot(np.array([dc, dr, ds]))
#    print(dD,[dc,dr,ds])
#    print('t',t)
    respone = I[x,y] + dg * 0.5
#    print(I[x,y],dg * 0.5,I[x,y]*2/dg)
    if np.abs(respone) * n < 0.04 *255:
        return [None]*4


    # 利用Hessian矩阵的迹和行列式计算主曲率的比值
    Tr = Dxx + Dyy
    det = Dxx * Dyy - Dxy * Dxy
    if det<=0 or Tr * Tr / det >= 5.0:
        return [None]*4

    Key_point = []
    Key_point.append(int((x + dr) * 2**o))
    Key_point.append(int((y + dc) * 2**o))
#    Key_point.append(o + (s << 8) + (int(np.round((ds + 0.5)) * 255) << 16))
#    Key_point.append(sigma * np.power(2.0, (s + ds) / n)*(1 << o) * 2)

    return Key_point,x,y,s



    
def adjust(DoG,o,s,x,y,contrastThreshold,edgeThreshold,sigma,n,SIFT_FIXPT_SCALE):
    SIFT_MAX_INTERP_STEPS = 5
    BORDER = 5

    point = []

    img_scale = 1.0 / (255 * SIFT_FIXPT_SCALE)
    deriv_scale = img_scale * 0.5
    second_deriv_scale = img_scale
    cross_deriv_scale = img_scale * 0.25

    img = DoG[o][s]
    i = 0
    while i < SIFT_MAX_INTERP_STEPS:
        if s < 1 or s > n or y < BORDER or y >= img.shape[1] - BORDER or x < BORDER or x >= img.shape[0] - BORDER:
            return None,None,None,None

        img = DoG[o][s]
        I_prev = DoG[o][s - 1]
        I_next = DoG[o][s + 1]

        dD = [ (img[x,y + 1] - img[x, y - 1]) * deriv_scale,
               (img[x + 1, y] - img[x - 1, y]) * deriv_scale,
               (I_next[x, y] - I_prev[x, y]) * deriv_scale ]

        v2 = img[x, y] * 2
        dxx = (img[x, y + 1] + img[x, y - 1] - v2) * second_deriv_scale
        dyy = (img[x + 1, y] + img[x - 1, y] - v2) * second_deriv_scale
        dss = (I_next[x, y] + I_prev[x, y] - v2) * second_deriv_scale
        dxy = (img[x + 1, y + 1] - img[x + 1, y - 1] - img[x - 1, y + 1] + img[x - 1, y - 1]) * cross_deriv_scale
        dxs = (I_next[x, y + 1] - I_next[x, y - 1] - I_prev[x, y + 1] + I_prev[x, y - 1]) * cross_deriv_scale
        dys = (I_next[x + 1, y] - I_next[x - 1, y] - I_prev[x + 1, y] + I_prev[x - 1, y]) * cross_deriv_scale

        H=[ [dxx, dxy, dxs],
            [dxy, dyy, dys],
            [dxs, dys, dss]]

        X = np.matmul(np.linalg.pinv(np.array(H)),np.array(dD))

        xi = -X[2]
        xr = -X[1]
        xc = -X[0]

        if np.abs(xi) < 0.5 and np.abs(xr) < 0.5 and np.abs(xc) < 0.5:
            break

        y += int(np.round(xc))
        x += int(np.round(xr))
        s += int(np.round(xi))

        i+=1

    if i >= SIFT_MAX_INTERP_STEPS:
        return None,x,y,s
    if s < 1 or s > n or y < BORDER or y >= img.shape[1] - BORDER or x < BORDER or x >= \
            img.shape[0] - BORDER:
        return None, None, None, None


    t = (np.array(dD)).dot(np.array([xc, xr, xi]))

    contr = img[x,y] * img_scale + t * 0.5
    if np.abs( contr) * n < contrastThreshold*255:
        return None,x,y,s


    # 利用Hessian矩阵的迹和行列式计算主曲率的比值
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    if det <= 0 or tr * tr * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * det:
        return None,x,y,s

    point.append((x + xr) * (1 << o))
    point.append((y + xc) * (1 << o))
    
#    point.append(o + (s << 8) + (int(np.round((xi + 0.5)) * 255) << 16))
#    point.append(sigma * np.power(2.0, (s + xi) / n)*(1 << o) * 2)

    return point,x,y,s










if __name__=='__main__':
    
    im1_path = r'C:\Users\Lenovo\Desktop\flowergray.jpg'
#    im1_path = r'C:\Users\Lenovo\Desktop\building1.jpg'
#    im1_path = r'C:\Users\Lenovo\Desktop\CARTOON.jpg'
    
    
    im1 = pli.imread(im1_path)
#    K = Guassian_Kernel(1.60,5)
#    p,D = generate_DOG(im1,6,sigma0=1.5,Octave_num=5)
    im2 = im1.copy()
    path1 = r'C:\Users\Lenovo\Desktop\pickle\building1_GP.pkl'
    path2 = r'C:\Users\Lenovo\Desktop\pickle\building1_DOG.pkl'
    
    path1 = r'C:\Users\Lenovo\Desktop\大学\大三AI\图像处理基础\图像处理编程作业2\图像处理练习2\G_P.pkl'
    path2 = r'C:\Users\Lenovo\Desktop\大学\大三AI\图像处理基础\图像处理编程作业2\图像处理练习2\DOG.pkl'
  
    
    with open(path1,'rb') as f1:
        GP = pickle.load(f1)
    with open(path2,'rb') as f2:
        DOG = pickle.load(f2)

    kpp = Extreme_points(DOG)
    
    for i,j in kpp:
        im2[int(i),int(j)]= 255

    show(im2,gray=1)

#    Make_pickle(r'C:\Users\Lenovo\Desktop\building1.jpg','building_GP','building_DOG')


