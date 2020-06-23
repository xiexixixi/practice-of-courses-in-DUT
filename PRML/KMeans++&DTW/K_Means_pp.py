# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:28:43 2020

@author: Lenovo
"""
import numpy as np
import matplotlib.pyplot as plt


class KMeans():
    def __init__(self,data):
        self.k = None
        self.data = np.array(data)
        self.sample_num = len(self.data)
        self.dim = self.data.shape[1]
        self.centers = []
        self.init_c = []
        self.clas = None
    def init_center(self,k):
        self.k = k
        self.centers = [self.data[0]]
        idxes = np.random.permutation(self.sample_num)
        for j in range(self.k-1):
            d = np.zeros((self.sample_num,len(self.centers)),dtype = np.int64)
            for i in range(len(self.centers)):
                d[:,i] = np.sum((self.data - self.centers[i])**2,axis = 1) 
            dist2 = d.min(axis=1)
            sum_dist2 = dist2.sum()
            for i in range(self.sample_num):
                idx = idxes[i]
                sum_dist2-=dist2[idx]
                if sum_dist2<=0:
                    self.centers.append(self.data[idx])
                    break     
        return
    
    def _update_center(self):
        update = []
        for i in range(self.k):
            cluster = self.data[self.clas==i]
            update.append(cluster.mean(axis=0))
        
        return update
        
        
    def _stop_updating(self,update):
        for i in range(self.k):
            if (update[i] != self.centers[i]).any():
                return False
        else:
            return True
    
    def kmeans(self,k,stop_iter = 100):
        self.k = k
        self.init_center(k)
        self.init_c = self.centers.copy()
        counter = 0
        while(counter<stop_iter):
            mat = np.zeros((self.sample_num,k))
            for i in range(k):
                mat[:,i] = np.sum((self.data - self.centers[i])**2,axis = 1)
            self.clas = mat.argmin(axis=1)
            
            update = self._update_center()
            if self._stop_updating(update):
                break
            else:
                self.centers = update
            counter+=1
        print("%d times update"%counter)
        return
    
    def predict(self,point):
        point = np.array(point)
        idx = np.argmin(np.sum((self.data-point)**2,axis=1))
        return self.clas[idx]
    def predict_seq(self,seq):
        return np.array([self.predict(point[1:]) for point in seq],dtype=np.int8)
        
    
    def show_centers(self,all_data = False):
        u = np.array(self.centers)
        if all_data:
            plt.scatter(self.data[:,0],self.data[:,1],marker='.')
        plt.scatter(u[:,0],u[:,1],marker='o',s=50,label='initial_center')
        return
    
    #输出聚类结果可视化
    def show_clusters(self):
        for i in range(self.k):
            cluster = self.data[self.clas==i]
            plt.scatter(cluster[:,0],cluster[:,1],marker='.',label='class%d'%(i+1))
        u = np.array(self.centers)
        init = np.array(self.init_c)
        plt.scatter(init[:,0],init[:,1],marker='^',label ='initial center')
        plt.scatter(u[:,0],u[:,1],marker='*',s=300,label ='center')
        plt.legend()
        
        return
    
    
if __name__ == '__main__':
    #聚类展示
    K = 5
    all_train_coord = np.array(get_train_coord(),dtype=np.int64)
    
    km = KMeans(all_train_coord)
    km.kmeans(K)
    km.show_clusters()

    
    
    
    
    
    
    
    
    
    
    
    