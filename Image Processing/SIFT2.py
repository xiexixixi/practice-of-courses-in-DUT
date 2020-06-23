# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:54:40 2020

@author: Lenovo
"""

from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pli
import pickle
from PIL import Image

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
    
    print(X,Angle)
    Y = Xrot(X,Angle)
    print(Y)
    scale_x = int(np.floor(Y[0].min())),int(np.ceil(Y[0].max()))
    scale_y = int(np.floor(Y[1].min())),int(np.ceil(Y[1].max()))
    
    x_range,y_range = scale_x[1]-scale_x[0],scale_y[1]-scale_y[0]
    
    x = np.linspace(x_range/2-0.5,-x_range/2+0.5,x_range)
    y = np.linspace(-y_range/2+0.5,y_range/2-0.5,y_range)
    
    new_xcoord,new_ycoord = np.meshgrid(-x,-y)
    
    
    
    grid_z = griddata(Y.T, lvector/255, (new_xcoord,new_ycoord), method='linear')
#    grid_z = mlab.griddata(new_xcoord.flatten(),new_ycoord.flatten(),lvector,)

    return grid_z
    

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
            
            
def spot(im3,kps,mds):
    for i,j in kps:
        if mds[kps.index([i,j])]==0 or mds[kps.index([i,j])]==45:
            im3[int(i),int(j)]= [255,0,0]
        elif mds[kps.index([i,j])]==180 or mds[kps.index([i,j])] == 225:
            im3[int(i),int(j)]= [0,255,0]
        elif mds[kps.index([i,j])]==90 or mds[kps.index([i,j])] == 135:
            im3[int(i),int(j)]= [0,0,255]
        else:
            im3[int(i),int(j)]= [255,255,255]
    show(im3)
    
    
def load_pickle(GP_file,DOG_file=None):
    
    path = r'C:\Users\Lenovo\Desktop\pickle\\'
    with open(path+GP_file+'.pkl','rb') as f:
        GP =pickle.load(f)
    if DOG_file == None:
        return GP
    with open(path+DOG_file+'.pkl','rb') as f:
        DOG =pickle.load(f)
    return GP,DOG


def preprocess(img):
    
    if img.ndim==3:
        img = RGB2Gray(img)
        
    target_size = (2**int(np.log2(min(img.shape))),2**int(np.log2(min(img.shape))))
    print(img.shape)
    i = imresize(img,target_size).copy()
    
    ratio_x = img.shape[0]/i.shape[0]
    ratio_y = img.shape[1]/i.shape[1]
    
    return i,(ratio_x,ratio_y)
        
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
    

def generate_DOG(I,Scale_num=6,sigma0=1.5,Octave_num=None):
    
    
    I,ratio = preprocess(I)
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





im1_path = r'C:\Users\Lenovo\Desktop\flowergray.jpg'
im1 = pli.imread(im1_path)


#p,D = generate_DOG(im1,6,sigma0=1.5,Octave_num=5)

#with open(r'C:\Users\Lenovo\Desktop\pickle\GP.pkl','wb') as f:
#    pickle.dump(p,f)
#    
#with open(r'C:\Users\Lenovo\Desktop\pickle\DOG.pkl','wb') as f:
#    pickle.dump(D,f)   


def Adjust(DOG,o,s,x,y):
    
    n = 3
    img_border = 5
    I = DOG[o][s].astype(float)
    for i in range(5):
        if s < 1 or s > n or y < img_border or y >= I.shape[1] - img_border or x < img_border or x >= I.shape[0] - img_border:
            return [None]*4
        
        I_prev = DOG[o][s-1].astype(np.float32)
        I = DOG[o][int(s)].astype(np.float32)
        I_next = DOG[o][int(s+1)].astype(np.float32)
        

        dD = np.array([I[x,y+1] - I[x,y-1],
                       I[x+1,y] - I[x-1,y],
                       I_next[x,y] - I_prev[x,y]],dtype=np.float32)*0.5
        
    
        v2 = I[x,y] * 2
        Dxx = (I[x,y+1] + I[x,y-1] - v2)
        Dyy = (I[x+1,y] + I[x-1,y] - v2)
        Dss = (I_next[x,y] + I_prev[x,y] - v2)
        Dxy = (I[x+1,y+1] - I[x+1,y-1] - I[x-1,y+1] + I[x-1,y-1]) * 0.25
        Dxs = (I_next[x,y+1] - I_next[x,y-1] - I_prev[x,y+1] + I_prev[x,y-1]) * 0.25
        Dys = (I_next[x+1,y] - I_next[x-1,y] - I_prev[x+1,y] + I_prev[x-1,y]) * 0.25
        H=np.array([[Dxx, Dxy, Dxs],
                    [Dxy, Dyy, Dys],
                    [Dxs, Dys, Dss]],dtype=np.float32)

        X = -np.matmul(np.linalg.pinv(H),dD)
        dx,dy,ds = X
        if (np.abs(X) < 0.5).all():
            break
        
        x+=int(round(dx))
        y+=int(round(dy))
        s+=int(round(ds))
        
    #迭代 5 次都没找到，丢弃
    else:
        return [None]*4
            
    #判断找到的点是否在边界内，边界外舍去
    if s < 1 or s > n or y < img_border or y >= I.shape[1] - img_border or x < img_border or x >= \
            I.shape[0] - img_border:
        return [None]*4
    
    #判断是可能是噪声，可能是噪声舍去
    dg = dD.dot(np.array([dx, dy, ds]))
    respone = I[x,y] + dg * 0.5
    if np.abs(respone) * n < 0.04 *255:
        return [None]*4
    
    # 利用Hessian矩阵的迹和行列式计算主曲率的比值
    Tr = Dxx + Dyy
    det = Dxx * Dyy - Dxy * Dxy
    if det<=0 or Tr * Tr / det >= 12.1:
        return [None]*4

    Key_point = []
    Key_point.append(int((x + dx) * 2**o))
    Key_point.append(int((y + dy) * 2**o))
        
    return x,y,s,Key_point


def MainPoint(GP_layer,o,s,r,c):
    
    
    sigma_oct = 1.52*2**(s/3+o)
    
    radius = int(np.ceil(3* 1.5*sigma_oct))
    
    dim = 2*radius+1
    K = Guassian_Kernel(1.5*sigma_oct,dim)
    dx = np.zeros((2*radius+1,2*radius+1),dtype=np.float32)
    dy = np.zeros((2*radius+1,2*radius+1),dtype=np.float32)
    sita = np.full_like(dx,-4.)
    for i in range(-radius,radius+1):
        x = r+i
        if x<1 or x>GP_layer.shape[0]-2:
            continue
        
        for j in range(-radius,radius+1):
            y = c+j
            if y<1 or y>GP_layer.shape[0]-2:
                continue
            dx[i+radius,j+radius] = GP_layer[x+1,y].astype(np.float32) - GP_layer[x-1,y].astype(np.float32)
            dy[i+radius,j+radius] = GP_layer[x,y+1].astype(np.float32) - GP_layer[x,y-1].astype(np.float32)
            
            sita[i+radius,j+radius] = np.arctan2(dy[i+radius,j+radius],dx[i+radius,j+radius])
    
    grad = (dx**2 + dy**2)**0.5 * K
    sita = sita*180/np.pi +180
    
    #统计
    hist = np.array([0.]*8)
    for i in range(2*radius+1):
        for j in range(2*radius+1):
            if sita[i,j]<0:
                continue
            
            hist[int((sita[i,j])//45.1)] +=grad[i,j]
    
    main_dir = np.argmax(hist)*45

    return main_dir

def Key_points_info(GP,DOG):
    
    kps=[]
    mds=[]
    kps_info = []
    O = len(DOG)
    S = len(DOG[0])
    boarder = 5
    for o in range(O):
        for s in range(1,S-1):
            I_pre, I_cur,I_next = DOG[o][s-1],DOG[o][s],DOG[o][s+1]
            stride = max(1,min(2,2**(int(np.log2(min(I_cur.shape)))-6)))
            
            print(o,s)
            for i in range(boarder,I_cur.shape[0]-boarder,stride):
                for j in range(boarder,I_cur.shape[1]-boarder,stride):
                    val = I_cur[i,j]
                    eight_neiborhood_prev = I_pre[i-1:i+2,j-1:j+2]
                    eight_neiborhood = I_cur[i-1:i+2,j-1:j+2]
                    eight_neiborhood_next = I_next[i-1:i+2,j-1:j+2]
    
                    if ((val > 0 and (val >= eight_neiborhood_prev).all() and (val >= eight_neiborhood).all() and (val >= eight_neiborhood_next).all())
                         or (val < 0 and (val <= eight_neiborhood_prev).all() and (val <= eight_neiborhood).all() and (val <= eight_neiborhood_next).all())):
                        
                        x,y,scale,kp = Adjust(DOG,o,s,i,j)
                        if kp == None:
                            continue
                        md = MainPoint(GP[o][scale],o,s,x,y)
                        kps.append(kp)
                        mds.append(md)
                        kps_info.append([o,scale,x,y,md,kp])
                                          
    return kps_info



def calcSIFTDescriptor(img,x,y,direction,radius,d=4,n=8):
    
    radius = 8
    radius = int(radius)
    features = []
#    cos_t = np.cos(direction * (np.pi / 180)) # 余弦值
#    sin_t = np.sin(direction * (np.pi / 180)) # 正弦值
    
    rang = 2*int(np.ceil(1.415*radius)+1)
    print('x,y',x,y)
    I = img[x-rang:x+rang,y-rang:y+rang]
    if I.shape[0]!=I.shape[1]:
        return None
    
    print(I.shape)    
    rot_I = Irot(I,-direction)
    x,y = rot_I.shape[0]//2,rot_I.shape[1]//2
    
    
    for r in range(-2,2):
        for c in range(-2,2):
            
            hist = [0.]*8
            for i in range(x+r*radius,x+r*radius+radius):
                for j in range(y+c*radius,y+c*radius+radius):
                    
                    dx = rot_I[i+1,j].astype(np.float32) - rot_I[i-1,j].astype(np.float32)
                    dy = rot_I[i,j+1].astype(np.float32) - rot_I[i,j-1].astype(np.float32)
                    theta = np.arctan2(dy,dx)*180/np.pi+180
                    grad = (dx**2 + dy**2)**0.5
                    
                    hist[int((theta)//45.1)] +=grad
                    
            features.append(hist)

    
    features = np.array(features,dtype='float32').flatten()
    
    
    nrm2 = 0
    length = d * d * n
    for k in range(length):
        nrm2 += features[k] * features[k]
    thr = np.sqrt(nrm2) * 0.2

    nrm2 = 0
    for i in range(length):
        val = min(features[i], thr)
        features[i] = val
        nrm2 += val * val
    nrm2 = 512 / max(np.sqrt(nrm2), 1.19209290E-07)
    
    for k in range(length):
        features[k] = min(max(features[k] * nrm2,0),255)

    return features



class Stitcher:
    def stitch(self,images,ratio=0.75,reprojThresh=4.0,showMatches=False):
        (imageB,imageA) = images
        
        
        (kpsA,featuresA) = self.detectAndDescribe2(imageA)
        (kpsB,featuresB) = self.detectAndDescribe2(imageB)
        
        M = self.matchKeypoints(kpsA,kpsB,featuresA,featuresB,ratio,reprojThresh)
        if M is None:
            return None
        
        (matches,H,status) = M
        
        result = cv2.warpPerspective(imageA,H,(imageA.shape[1]+imageB.shape[1]))
        self.cv_show('result',result)
        
        result[0:imageB.shape[0],0:imageB.shape[1]] = imageB
        self.cv_show('result',result)
        
        if showMatches:
            vis = self.drawMatches(imageA,imageB,kpsA,kpsB,matches,status)
            
        
        
    def detectAndDescribe(self,image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        descriptor = cv2.xfeatures2d.SIFT_create()
        kps,features = descriptor.detectAndCompute(image,None)
        kps =np.float32([kp.pt for kp in kps])
        
        return kps,features
    
    def detectAndDescribe2(self,image):
        GP,DOG = generate_DOG(image)
        kps_info = Key_points_info(GP,DOG)
        location = [kp[0:4] for kp in kps_info]
        mds = [kp[4] for kp in kps_info]
        kps = [kp[5] for kp in kps_info]
        descriptors = []
        for i in range(len(kps_info)):
            o,s,x,y = location[i]
            md = mds[i]
            sigma_oct = 1.52*2**(s/3+o)
        
            radius = int(np.ceil(3* 1.5*sigma_oct))
            descriptors.append(calcSIFTDescriptor(GP[o][s],x,y,md,radius))  
            
        return kps,descriptors     
    
    
    
    def matchKeypoints(self,kpsA,kpsB,featuresA,featuresB,ratio,reprojThresh):
        
        
        
        matcher = cv2.BFMatcher()
        
        rawMatches = matcher.knnMatch(featuresA,featuresB,2)
        
        matches = []
    
        
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance*ratio:
                matches.append((m[0].trainIdx,m[0].queryIdx))
                
        if len(matches)>4:
            ptsA = np.float32([kpsA[i] for (_,i) in matches])
            ptsB = np.float32([kpsB[i] for (i,_) in matches])
            
            H,status = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,reprojThresh)
            
            return matches,H,status
        
        

def detectAndDescribe2(image):
    GP,DOG = generate_DOG(image)
    kps_info = Key_points_info(GP,DOG)
    location = [kp[0:4] for kp in kps_info]
    mds = [kp[4] for kp in kps_info]
    kps = [kp[5] for kp in kps_info]
    descriptors = []
    kps_reduce = []
    for i in range(len(kps_info)):
        o,s,x,y = location[i]
        md = mds[i]
        sigma_oct = 1.52*2**(s/3+o)
    
        radius = int(np.ceil(3* 1.5*sigma_oct))
        f = calcSIFTDescriptor(GP[o][s],x,y,md,radius)
        if f == None:
            continue
        
        kps_reduce.append(kps[i])
        descriptors.append(calcSIFTDescriptor(GP[o][s],x,y,md,radius))  
        
        
        
    return kps_reduce,descriptors


def kps_feature(pic_name):
    
    kps_info = []
    feature = []
    
    with open(r'C:\Users\Lenovo\Desktop\pickle\\'+pic_name+'_kps_info.pkl','wb') as f:
        kps_info = pickle.load(f)

        
    with open(r'C:\Users\Lenovo\Desktop\pickle\\'+pic_name+'_descriptors.pkl','wb') as f:
        feature = pickle.load(f)

    kps = [kp[5] for kp in kps_info]

    return kps,feature


GP,DOG = load_pickle('building1_GP','building1_DOG')

kps_info = Key_points_info(GP,DOG)

#with open(r'C:\Users\Lenovo\Desktop\pickle\building1_kps_info.pkl','wb') as f:
#    pickle.dump(kps_info,f)


#kps_info = load_pickle('building1_kps_info')

location = [kp[0:4] for kp in kps_info]
mds = [kp[4] for kp in kps_info]
kps = [kp[5] for kp in kps_info]



im1_path = r'C:\Users\Lenovo\Desktop\building1.jpg'
im1 = pli.imread(im1_path)

im2 = imresize(im1,(1024,1024)).copy()

for i,j in kps:
    spot(im2,kps,mds)


#sigma = 1.6
#d = 4
#kps_remove=[]
#descriptors = []
#for i in range(len(kps_info)):
#    o,s,x,y = location[i]
#    md = mds[i]
#    sigma_oct = 1.52*2**(s/3+o)
#
#    radius = int(np.ceil(3* 1.5*sigma_oct))
#    
#    f = calcSIFTDescriptor(GP[o][s],x,y,md,radius)
#    if f is None:
#        continue
#    kps_remove.append(kps[i])
#    descriptors.append(f)
#    
#
#
#
#im1_path = r'C:\Users\Lenovo\Desktop\building1.jpg'
#im1 = pli.imread(im1_path)    




#im2 = im1.copy()
#if im2.ndim==3:
#    im2 = RGB2Gray(im2)
#    im3 = imresize(im2,(1024,1024)).copy()
#
#for i,j in kps:
#    im3[int(i),int(j)]= 255
#show(im3,gray=1)


         





