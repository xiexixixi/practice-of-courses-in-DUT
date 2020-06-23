# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:30:02 2020

@author: Lenovo
"""

from scipy.interpolate import griddata
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pli

im1_path = r'C:\Users\Lenovo\Desktop\flowergray.jpg'
im1 = pli.imread(im1_path)

def Xrot(X, angle):
    anglePi = angle * np.pi / 180.0
    cos = np.cos(anglePi)
    sin = np.sin(anglePi)
    Rot_M = np.array([[cos,-sin],
                      [sin, cos]])
    Y = np.matmul(Rot_M,X)
    
    return Y

def Irot(Image,Angle):
    h = Image.shape[0]
    w = Image.shape[1]
    
    x = np.linspace(h/2-0.5,-h/2+0.5,h)
    y = np.linspace(-w/2+0.5,w/2-0.5,w)
    xcoord,ycoord = np.meshgrid(y,x)
    
    X = np.concatenate((xcoord.flatten()[np.newaxis,:],ycoord.flatten()[np.newaxis,:]),axis = 0)
    
    lvector = Image.reshape(Image.shape[0]*Image.shape[1])
    
    Y = Xrot(X,Angle)
    scale_x = int(np.floor(Y[0].min())),int(np.ceil(Y[0].max()))
    scale_y = int(np.floor(Y[1].min())),int(np.ceil(Y[1].max()))
    
    x_range,y_range = scale_x[1]-scale_x[0],scale_y[1]-scale_y[0]
    
    x = np.linspace(x_range/2-0.5,-x_range/2+0.5,x_range)
    y = np.linspace(-y_range/2+0.5,y_range/2-0.5,y_range)
    
    new_xcoord,new_ycoord = np.meshgrid(-x,-y)
    
    
    
    grid_z = griddata(Y.T, lvector, (new_xcoord,new_ycoord), method='linear')
    #grid_z = mlab.griddata(new_xcoord.flatten(),new_ycoord.flatten(),lvector,)

    plt.imshow(grid_z)
    return grid_z
    


def show_rot(Image):
    I = Image.copy()
    #while True:
    plt.imshow(I)
    pos = plt.ginput(2)
    angle = np.arctan((pos[0][1]-pos[1][1])/(pos[0][0]-pos[1][0]))
    print(angle)
    angle=angle*180/np.pi
    print(angle)

    
    if((pos[0]<pos[1]).all() or (pos[0]>pos[1]).all()):
        Irot(I,angle)
    else:
        Irot(I,-angle)
        
#rot(im1,60)
show_rot(im1)





