# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:55:47 2020

@author: Lenovo
"""


import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as pli

def imresize(original_image,target_size):

    i = Image.fromarray(original_image)
    ii = i.resize(target_size[::-1],Image.BILINEAR)
    image = np.asarray(ii)
    return image

def downsampling(data,stride):
    return data[::stride,::stride]

def Convolve(I):
    F = np.array([[1/9]*9]).reshape(3,3)
    iw,ih = I.shape
    fw,fh = F.shape
    Conv_I = np.zeros_like(I)
    Image_pad = np.pad(I,(((fh-1)//2,(fh-1)//2),((fw-1)//2,(fw-1)//2)), 'reflect')
    print(Image_pad.shape)
    func = lambda x,y:np.sum(Image_pad[x:x+fh,y:y+fw]*F)
    for i in range(iw):
        for j in range(ih):
            Conv_I[i,j] = func(i,j)
    return Conv_I

def PCA_info(data):
    samples,dimensions = data.shape
    m = data.mean(axis=0)
    z = (data-m).astype(np.float32)
    S = np.matmul(z.T,z)/samples
    
    eigvals,eigvec=np.linalg.eigh(S)
    
    eigvec = eigvec[:,eigvals.argsort()[::-1]]
    eigvals = eigvals[eigvals.argsort()[::-1]]
    return m,eigvec,eigvals
    

def PCA(data,eigvec,m,n_components):
    
    '''输入数据，特征向量，均值，降的维度——输出降维后的特征'''
    M = eigvec[:,:n_components]
    z = (data-m).astype(np.float32)
    return np.matmul(z,M)


def get_data_set(downsampling = 8):
    Subjects = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    file_list = os.listdir('.\\project1_data_Recognition')
    for pics_name in file_list:
        
        if pics_name[-3:] != 'pgm':
            continue
        pic = pli.imread(os.path.join('.\\project1_data_Recognition',pics_name))
        pic = pic[::downsampling,::downsampling]
        Subjects[int(pics_name[7:9])-1].append(pic)
        
    return Subjects


def get_train_test_data(Subjects,N=3):
    
    
    randperm = np.random.permutation(np.arange(11))
    train_ind = randperm[:N]
    test_ind = randperm[N:]
    
    if type(Subjects) == list:
        Subs = np.array(Subjects)
    Subs = Subjects.copy()
  
    shape = Subs.shape
    
    train_set =  Subs[:,train_ind,:,:].reshape(15*N,shape[2]*shape[3])
    test_set = Subs[:,test_ind,:,:].reshape(15*(11-N),shape[2]*shape[3])
    
    return train_set,test_set


#分类,计算错误率
def classify_accuracy(redu_data,test_set,N):
    close_ind = []
    wrong = 0
    for i in range(len(test_set)):
        test = test_set[i]
        
        distances = np.square(redu_data-test).sum(axis=1)
        min_dis_ind = np.argmin(distances)
        
        close_ind.append(min_dis_ind//N)
        if min_dis_ind//N != i//(11-N):
            wrong+=1
        
    misrate = wrong/len(test_set)

    return close_ind,misrate

def exc_experiment_PCA(Subjects,N,K=None):
    
    
    train_set,test_set = get_train_test_data(Subjects,N)
    print(train_set.shape,test_set.shape)
    m,eigvec,eigvals = PCA_info(train_set)
    
    
    if K is None:
        K = np.arange(0,100,10)
    
    misrates = []
    for n_components in K:
        redu_train = PCA(train_set,eigvec,m,n_components)
        redu_test =  PCA(test_set,eigvec,m,n_components)
        close_ind,misrate = classify_accuracy(redu_train,redu_test,N)
        
        misrates.append(misrate)
    
    return np.array(misrates),m


def exc_experiment_PCA_10_times(Subjects,N,K=None):
    
    misrates = np.zeros(len(K),dtype=np.float32)
    for _ in range(10):
        misrate,_ = exc_experiment_PCA(Subjects,N,K)
        misrates += misrate
    
    
    return misrates/10


def get_eigen(train_set,test_set,N):
    m,eigvec,eigvals = PCA_info(train_set)
    
    redu_train = PCA(train_set,eigvec,m,15*N-15)
    redu_test =  PCA(test_set,eigvec,m,15*N-15)
    
    mi = redu_train.reshape(N,15,15*N-15)
    mi = mi.mean(axis=0)
    m = redu_train.mean(axis=0)
    
    
    Sw = np.zeros((15*N-15,15*N-15),dtype=np.float64)
    for i in range(15):
        Xi = redu_train[i*N:(i+1)*N] - mi[i]
        Si = np.matmul(Xi.T,Xi).astype(np.float64)
        Sw+=Si
    
    
    ma = mi-m
    Sb = 15*N*np.matmul(ma.T,ma)    
    

    Mat = np.matmul(np.linalg.inv(Sw),Sb)
    eigvals,eigvec = np.linalg.eig(Mat)
    eigvec = eigvec[:,eigvals.argsort()[::-1]]
    
    return redu_train,redu_test,eigvec
    


def exc_experiment_LDA_(Subjects,N,K):
    train_set,test_set = get_train_test_data(Subjects,N)
    redu_train,redu_test,eigvec = get_eigen(train_set,test_set,N)
    
    print(redu_train.shape)
    misrates = []
    K = min(K,14)
    for n_components in range(1,K):
        M = eigvec[:,:n_components]
        train_features = np.matmul(redu_train,M)
        test_features = np.matmul(redu_test,M)
        close_ind,misrate = classify_accuracy(train_features,test_features,N)
        
        misrates.append(misrate)

    return np.array(misrates)


def exc_experiment_LDA(Subjects,N,K):    
    train_set,test_set = get_train_test_data(Subjects,N)
    m,eigvec,eigvals = PCA_info(train_set)
    
    redu_train = PCA(train_set,eigvec,m,15*N-15)
    redu_test =  PCA(test_set,eigvec,m,15*N-15)
    
    mi = redu_train.reshape(15,N,15*N-15)
    mi = mi.mean(axis=1)
    m = redu_train.mean(axis=0)
    
    ma = mi-m
    
    Sb = N*np.matmul(ma.T,ma)
    
    Sw = np.zeros((15*N-15,15*N-15),dtype=np.float64)
    for i in range(15):
        Xi = redu_train[i*N:(i+1)*N] - mi[i]
        Si = np.matmul(Xi.T,Xi).astype(np.float64)
        Sw+=Si
    
    
    Mat = np.matmul(np.linalg.inv(Sw),Sb)
    eigvals,eigvec = np.linalg.eig(Mat)
    
    eigvec = eigvec[:,eigvals.argsort()[::-1]]
    
    #获得训练集降维特征
    
    
    
    misrates = []
    for n_components in range(1,K+1):
        M = eigvec[:,:n_components]
        train_features = np.matmul(redu_train,M)
        test_features = np.matmul(redu_test,M)        

        close_ind,misrate = classify_accuracy(train_features,test_features,N)
        misrates.append(misrate)
    
    return np.array(misrates)


def exc_experiment_LDA_10_times(Subjects,N,K):
    

    misrates = np.zeros(K,dtype=np.float32)
    print(misrates.shape)
    for _ in range(10):
        misrates += exc_experiment_LDA(Subjects,N,K)
    
    
    return misrates/10


def LBP_features(m_face):
    
    LBP_face = np.full_like(m_face,4,dtype='uint8')
    for i in range(1,m_face.shape[0]-1):
        for j in range(1,m_face.shape[1]-1):
            center = m_face[i,j]
            code = 0
            code |= (m_face[i-1][j-1] >= center) << (np.uint8)(7)  
            code |= (m_face[i-1][j  ] >= center) << (np.uint8)(6)  
            code |= (m_face[i-1][j+1] >= center) << (np.uint8)(5)  
            code |= (m_face[i  ][j+1] >= center) << (np.uint8)(4)  
            code |= (m_face[i+1][j+1] >= center) << (np.uint8)(3)  
            code |= (m_face[i+1][j  ] >= center) << (np.uint8)(2)  
            code |= (m_face[i+1][j-1] >= center) << (np.uint8)(1)  
            code |= (m_face[i  ][j-1] >= center) << (np.uint8)(0)  
            LBP_face[i][j]= code
    return LBP_face


def Hamin_dis(x,y):
    x=int(x)
    y=int(y)
    ax = 10 - len(bin(x))
    ay = 10 - len(bin(y))
    
    bix = [eval(i) for i in (ax*'0'+bin(x)[2:])]
    biy = [eval(i) for i in (ay*'0'+bin(y)[2:])]
    
    dis = np.nonzero([bix[i]-biy[i] for i in range(len(bix))])
    
    return len(dis[0])


def Hamin_dis_btw_LBP(LBP1,LBP2):
    
    dis_face = np.zeros_like(LBP1,dtype='uint8')
    for i in range(1,dis_face.shape[0]-1):
        for j in range(1,dis_face.shape[1]-1):
            dis_face[i,j] = Hamin_dis(LBP1[i,j],LBP2[i,j])
            
    distance = sum(dis_face.sum(axis=-1))/LBP1.size
    return distance

class HOG(object):
    def __init__(self, image=None,cell_size = 8, bin_num = 9, block_shape=(2,2),gama = 1):
        self.cell_size = cell_size
        self.bshape = block_shape
        self.image_origen = image
        self.bin_num = bin_num
        self.gama = gama

        self.image = None
        self.dx = None
        self.dy = None
        self.Mag = None
        self.theta = None
        
        self.cells_feature = None
        self.HOG_feature = None
        self.cells_feature_calc = False
        self.block_feature_calc = False
    
    def clear(self):
        self.image = None
        self.HOG_features = None
        self.dx = None
        self.dy = None
        self.Mag = None
        self.theta = None
        
        self.cells_feature = None
        self.HOG_feature = None
        self.cells_feature_calc = False
        self.block_feature_calc = False
        return
    
    def set_info(self, cell_size = 6, bin_num = 9, block_shape=(2,2),image=None,gama = 1):
        self.cell_size = cell_size
        self.bshape = block_shape
        self.image_origen = image
        self.bin_num = bin_num
        self.gama = gama
        return
    def preprocess_image(self,image,gama = None):
        
        
        if gama is not None:
            self.gama = gama

        self.image_origen = image
        if self.image_origen.ndim == 3:
            self.image = self.image_origen.mean(axis=-1)
        else:
            self.image = image
        self.image = np.power(self.image/float(np.max(self.image)),self.gama)
        self.imshape = self.image.shape
        self.image_shape = image.shape
        return self.image
    
    def grad_info(self):
        Image_pad = np.pad(self.image,((1,1),(1,1)), 'reflect')
        self.dx = 0.5*(Image_pad[2:,:]-Image_pad[:-2,:])[:,1:-1]
        self.dy = 0.5*(Image_pad[:,2:]-Image_pad[:,:-2])[1:-1,:]
        self.Mag = np.sqrt(self.dx**2+self.dy**2)
        self.theta = np.arctan2(self.dy,self.dx)/np.pi*180 + 180 
        return

    def cell_feature(self,cell_M,cell_theta,bin_num):
        Hist = [0]*bin_num
        degree_per_bin = 360//bin_num
        for i in range(cell_M.shape[0]):
            for j in range(cell_M.shape[1]):
                min_axis = int(cell_theta[i,j]/degree_per_bin)%bin_num
                max_axis = int(cell_theta[i,j]/degree_per_bin+1)%bin_num
                mod_theta = cell_theta[i,j]%degree_per_bin
                
                Hist[min_axis] += cell_M[i,j]*(1-mod_theta/degree_per_bin)
                Hist[max_axis] += cell_M[i,j]*(mod_theta/degree_per_bin)
        
        return np.array(Hist)   
    
    def calc_cells_feature(self):
        self.cells_feature = np.zeros((self.imshape[0]//self.cell_size,self.imshape[1]//self.cell_size,self.bin_num),dtype=np.float32)
        for i in range(self.cells_feature.shape[0]):
            for j in range(self.cells_feature.shape[1]):
                
                cell_M = self.Mag[i*self.cell_size:(i+1)*self.cell_size,j*self.cell_size:(j+1)*self.cell_size]
                cell_theta = self.theta[i*self.cell_size:(i+1)*self.cell_size,j*self.cell_size:(j+1)*self.cell_size]
                
                self.cells_feature[i,j] = self.cell_feature(cell_M,cell_theta,self.bin_num)
    
    def calc_HOG_feature(self):
        
        if not self.block_feature_calc:
            bfshape = (self.cells_feature.shape[0]-self.bshape[0]+1,self.cells_feature.shape[1]-self.bshape[1]+1,self.bshape[0]*self.bshape[1]*self.bin_num)
            self.HOG_feature = np.zeros(bfshape)
            for i in range(self.cells_feature.shape[0]-1):
                for j in range(self.cells_feature.shape[1]-1):
                    block = self.cells_feature[i:i+self.bshape[0],j:j+self.bshape[1]].reshape(-1)
                    self.HOG_feature[i,j] = block / np.sum(block**2)**0.5   
            self.block_feature_calc = True
        return
    
    def get_HOG_feature(self,image=None,flatten=True):
        
        if image is not None:
            self.preprocess_image(image)
        if self.image is None:
            print('Please input image')
            return None

        self.grad_info()
        self.calc_cells_feature()
        self.calc_HOG_feature()
        
        if flatten:
            return self.HOG_feature.flatten()
        else:
            return self.HOG_feature
    
    def show(self,origen=False):
        if origen:
            if self.image_origen.ndim == 3:
                plt.imshow(self.image_origen)
            else:
                plt.imshow(self.image_origen,cmap='gray')
        else:
            plt.imshow(self.image,cmap='gray')
        return
    
    def info(self):
        print('cell_size:\t%d'%(self.cell_size))
        print('block shape:\t{}cells'.format(self.bshape))
        print('bin_num:\t%d'%self.bin_num)
        print('gama:\t\t%f'%self.gama)


class Detector(object):
    def __init__(self,image,subject = None,cell_size = 8, bin_num = 9, block_shape=(2,2),gama = 1):
        
        if subject is None:
            self.subject = HOG(cell_size = 8, bin_num = 9, block_shape=(2,2),gama = 1)
        else:
            self.subject = subject
            
        self.origen_image = image
        if self.origen_image.ndim == 3:
            self.image = self.origen_image.mean(axis=-1) 
        else:
            self.image = self.origen_image
        
        self.x_seg = None
        self.y_seg = None
        self.x_div = None
        self.detect_shape = None
    
        
        self.rang = []
        self.coords = []
        self.distances = []
        
        self.distances_LBP =[]
        self.coords_LBP = []
        self.rang_LBP = []
        
    def calc_subj_feature(self,subj,gama=None):

        self.subject.preprocess_image(subj,gama)
        self.subject.get_HOG_feature(subj,flatten=True)
        
        self.detect_shape = self.subject.image.shape
        return
    
    def set_detector_info(self,area_range = (70,60),div = (2,2)):
        
        self.x_range,self.y_range = area_range
        self.x_div,self.y_div = div
        
        self.x_stride = self.x_range // self.x_div
        self.y_stride = self.y_range // self.y_div


    
    def detect_HOG(self,gama=3.0,downratio = 1.5):
        
        Sub_HOG_feature = self.subject.HOG_feature.flatten()
        self.max_ds = int(np.log(min(self.image.shape[0]//self.x_range,self.image.shape[1]//self.y_range))/np.log(downratio))
        for downsampling_time in range(0,self.max_ds):
            
            print(downsampling_time,self.image.shape,self.image.max(),self.image.min())
            x_seg = self.image.shape[0] // self.x_range
            y_seg = self.image.shape[1] // self.y_range
            
            if x_seg<1 or y_seg<1:
                break
            for i in range((x_seg-1)*self.x_div):    
                for j in range((y_seg-1)*self.y_div):
                    detect_area = self.image[self.x_stride*i:self.x_stride*i+self.x_range,self.y_stride*j:self.y_stride*j+self.y_range]
                    detect_area = imresize(detect_area,self.detect_shape)
                    
                    hog_tool = HOG(gama)
                    HOG_features_ = hog_tool.get_HOG_feature(detect_area)
                    
                    distance = np.sum((HOG_features_-Sub_HOG_feature)**2)**0.5
                    
                    if str(distance) == 'nan':
                        break
                    self.distances.append(distance)
                    self.coords.append([int(self.x_stride*i*downratio),int(self.y_stride*j*downratio)])
                    self.rang.append([int(self.x_range*downratio**downsampling_time),int(self.y_range*downratio**downsampling_time)])
                
            self.image = imresize(self.image,(int(self.image.shape[0]/downratio),int(self.image.shape[1]/downratio)))
        
        self.coords=np.array(self.coords)
        self.rang = np.array(self.rang)
        self.distances = np.array(self.distances)
        
        
        self.coords = self.coords[np.argsort(self.distances)]
        self.rang = self.rang[np.argsort(self.distances)]
        self.distances = self.distances[np.argsort(self.distances)]
    
    def detect_LBP(self,downratio = 1.5):
        
        self.LBP_pic2 = LBP_features(self.image)
        
        self.LBP_face = LBP_features(self.subject.image)
        self.max_ds = int(np.log(min(self.image.shape[0]//self.x_range,self.image.shape[1]//self.y_range))/np.log(downratio))
        for downsampling_time in range(0,self.max_ds-1):
            
            print(downsampling_time,self.image.shape,self.image.max(),self.image.min())
            x_seg = self.image.shape[0] // self.x_range
            y_seg = self.image.shape[1] // self.y_range
            
            if x_seg<1 or y_seg<1:
                break
            for i in range((x_seg-1)*self.x_div):    
                for j in range((y_seg-1)*self.y_div):
                    
                    detect_area = self.LBP_pic2[self.x_stride*i:self.x_stride*i+self.x_range,self.y_stride*j:self.y_stride*j+self.y_range]
                    detect_area = imresize(detect_area,self.detect_shape)
                    
                    
                    self.distances_LBP.append(Hamin_dis_btw_LBP(detect_area,self.LBP_face))
                    self.coords_LBP.append([int(self.x_stride*i*downratio),int(self.y_stride*j*downratio)])
                    self.rang_LBP.append([int(self.x_range*downratio**downsampling_time),int(self.y_range*downratio**downsampling_time)])
                
            self.image = imresize(self.image,(int(self.image.shape[0]/downratio),int(self.image.shape[1]/downratio)))
        
        self.coords_LBP=np.array(self.coords_LBP)
        self.rang_LBP = np.array(self.rang_LBP)
        self.distances_LBP = np.array(self.distances_LBP)
        
        
        self.coords_LBP = self.coords_LBP[np.argsort(self.distances_LBP)]
        self.rang_LBP = self.rang_LBP[np.argsort(self.distances_LBP)]
        self.distances_LBP = self.distances_LBP[np.argsort(self.distances_LBP)]
        
        
        
    def show(self,n,method = 'HOG'):
        if method == 'HOG':
            x,y=self.coords[n]
            rx,ry = self.rang[n]
        else:
            x,y=self.coords_LBP[n]
            rx,ry = self.rang_LBP[n]
        plt.imshow(image[x:x+rx,y:y+ry],cmap='gray')
    
    
    def display_candidates(self,picnum=5,method = 'HOG',kill_fake = True):
        
        k = picnum//5
        
        coor = []
        ran = []
        j=0
        count = 0
    
        if kill_fake:
            while(count<=picnum):
                
                x,y=self.coords[j]
                rx,ry = self.rang[j]                   
                if x+rx >self.origen_image.shape[0] or y+ry>self.origen_image.shape[1]:
                    j+=1
                    continue
                if not self.is_fake_face(self.coords[j],self.rang[j]):
                    

                    coor.append(self.coords[j])
                    ran.append(self.rang[j])
                    count +=1
                j+=1
                            
            for i in range(picnum):
                if method == 'HOG':
    
                    x,y= coor[i]
                    rx,ry = ran[i]
                else:
                    x,y=self.coords_LBP[i]
                    rx,ry = self.rang_LBP[i]
                if x+rx <self.origen_image.shape[0] and y+ry<self.origen_image.shape[1]:

                    plt.subplot(k,5,i+1) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
                    plt.imshow(self.origen_image[x:x+rx,y:y+ry])
                    
        else:
            for i in range(picnum):
                if method == 'HOG':
    
                    x,y=self.coords[i]
                    rx,ry = self.rang[i]
                else:
                    x,y=self.coords_LBP[i]
                    rx,ry = self.rang_LBP[i]
                if x+rx <self.origen_image.shape[0] and y+ry<self.origen_image.shape[1]:
                    plt.subplot(k,5,i+1) 
                    plt.imshow(self.origen_image[x:x+rx,y:y+ry])
                
            
            
    def calc_IoU(self,c1,r1,c2,r2):
        
        rec1 = (c1[0],c1[1],c1[0]+r1[0],c1[1]+r1[1])
        rec2 = (c2[0],c2[1],c2[0]+r2[0],c2[1]+r2[1])

        
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    
        S = S_rec1 + S_rec2
    
        l = max(rec1[1], rec2[1])
        r = min(rec1[3], rec2[3])
        t = max(rec1[0], rec2[0])
        b = min(rec1[2], rec2[2])
    
        if l >= r or t >= b:
            return 0
        else:
            intersect = (r - l) * (b - t)
            return (intersect / (S - intersect))*1.0    
            
    def is_fake_face(self,c,r,seg_shape = (3,3),ratio_thresh=1.8):
        x,y = c
        rx,ry = r
        if x+rx >self.origen_image.shape[0] or y+ry>self.origen_image.shape[1]:
            return True

        
        sx,sy = r[0]//seg_shape[0], r[1]//seg_shape[1]
        
        if self.origen_image.ndim == 2:
            return True
        
        detected = self.origen_image[c[0]:c[0]+r[0],c[1]:c[1]+r[1]]
        RGB_hist = np.zeros((1,3),dtype = np.int64)
        RGB_ratio = []
        main_color = []
        for i in range(seg_shape[0]):
            for j in range(seg_shape[1]):
                area = detected[i*sx:(i+1)*sx,j*sy:(j+1)*sy]
                RGB_hist = area.sum(axis=(0,1))+1
                RGB_ratio.append(RGB_hist.max()/RGB_hist.min())
                main_color.append(RGB_hist.argmax())
                
        RGB_ratio = np.array(RGB_ratio)
        main_color = np.array(main_color)
        
        abnormal_num1 = len(RGB_ratio[RGB_ratio>ratio_thresh])
        abnormal_num2 = len(main_color[main_color != 0])
        
        seg_num = seg_shape[0]*seg_shape[1]
        
        
        return not (abnormal_num1*3 < seg_num and abnormal_num2*2 <seg_num)
                
    
    
        
    def select_face(self,candidates_num = 5,IoU_thre = 0.3):
        
        #根据IoU合并脸部，
        self.faces_info = [[self.coords[0],self.rang[0]]]
        for i in range(1,candidates_num):
            c = self.coords[i]
            r = self.rang[i]
            
            if self.is_fake_face(c,r):
                continue
            
            for j in range(len(self.faces_info)):
                fc = self.faces_info[j][0]
                fr = self.faces_info[j][1]
                #calc the IoU, judge if the face has been counted
                if self.calc_IoU(c,r,fc,fr)<IoU_thre:
                    continue

                else:
                    center1 = (c[0]+r[0]//2,c[1]+r[1]//2)
                    center2 = (fc[0]+fr[0]//2,fc[1]+fr[1]//2)
                    ran = [max(fr[0],r[0]),max(fr[1],r[1])]

                    coor = [(center1[0]+center2[0]-ran[0])//2,(center1[1]+center2[1]-ran[1])//2]
                    coor[0] = np.clip(coor[0],0,self.origen_image.shape[0]-ran[0])
                    coor[1] = np.clip(coor[1],0,self.origen_image.shape[1]-ran[1])

                    self.faces_info[j] = [np.array(coor),np.array(ran)]
                    break
                
            #是独一无二的脸部
            else:
                self.faces_info.append([c,r])
            
        return len(self.faces_info)
    
    def rect_face(self,width = 2):
        self.rect_face_image = self.origen_image.copy()
        if self.rect_face_image.ndim == 3:
            for c,r in self.faces_info:
                self.rect_face_image[c[0]:c[0]+r[0],c[1]:c[1]+width] = [255,0,0]
                self.rect_face_image[c[0]:c[0]+r[0],c[1]+r[1]-width:c[1]+r[1]] = [255,0,0]
                self.rect_face_image[c[0]:c[0]+width,c[1]:c[1]+r[1]] = [255,0,0]
                self.rect_face_image[c[0]+r[0]-width:c[0]+r[0],c[1]:c[1]+r[1]] = [255,0,0]
            
            
            plt.figure()
            plt.imshow(self.rect_face_image)
        else :
            for c,r in self.faces_info:
                self.rect_face_image[c[0]:c[0]+r[0],c[1]:c[1]+width] = 255
                self.rect_face_image[c[0]:c[0]+r[0],c[1]+r[1]-width:c[1]+r[1]] = 255
                self.rect_face_image[c[0]:c[0]+width,c[1]:c[1]+r[1]] = 255
                self.rect_face_image[c[0]+r[0]-width:c[0]+r[0],c[1]:c[1]+r[1]] = 255
            plt.figure()  
            plt.imshow(self.rect_face_image,cmap = 'gray')
            
  
if __name__ == '__main__':
    
    #获得要用的对象
    N = 11
    K=3
    Subjects = get_data_set(downsampling=6)
    Subjects = np.array(Subjects)
    shape = Subjects.shape
    train_set,test_set = get_train_test_data(Subjects,N)
    print(train_set.shape,test_set.shape)
    m_face = Subjects.reshape(Subjects.shape[0]*Subjects.shape[1],shape[2],shape[3]).mean(axis=0)
    
    
    
    
    image = pli.imread(r'.\project1_data_Detection\2.jpg')
    
    detector = Detector(image,cell_size = 6, bin_num = 9, block_shape=(2,2))
    detector.calc_subj_feature(m_face)
    detector.set_detector_info(area_range = (90,70),div = (3,2))
    detector.detect_HOG(gama = 3,downratio=1.1)
    
    detector.display_candidates(picnum = 15,method = 'HOG',kill_fake = True)
    
    faces = detector.select_face(4)
    
#    展示标出的人脸
    detector.rect_face()
    


