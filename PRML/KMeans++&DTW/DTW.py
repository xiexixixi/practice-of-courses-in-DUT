# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:39:26 2020

@author: Lenovo
"""

import numpy as np

class DTW(object):
    def __init__(self,seq1=[],seq2=[],calc_dis = None):
        self.seq1 = np.array(seq1)
        self.seq2 = np.array(seq2)
        
        self.s1_length = len(seq1)
        self.s2_length = len(seq2)
        
        self.path_map = []
        self.distance = 0
        if calc_dis is None:
            self.calc_dis = lambda X,Y:np.linalg.norm(np.array(X-Y),2)
        else:
            self.calc_dis = calc_dis
        self.accum_dis_mat = None
    def calc_distance(self,seq1=[],seq2=[],calc_dis = None):
        self.seq1 = np.array(seq1)
        self.seq2 = np.array(seq2)
        
        self.s1_length = len(seq1)
        self.s2_length = len(seq2)        
        if calc_dis is None:
            self.calc_dis = lambda X,Y:np.linalg.norm(np.array(X-Y),2)
        else:
            self.calc_dis = calc_dis
        self.distance = 0
        self.arg = np.zeros((self.s1_length,self.s2_length))
        dis_mat = np.zeros((self.s1_length,self.s2_length))

        
        
        for i in range(self.s1_length):
            for j in range(self.s2_length):
                dis_mat[i,j]=self.calc_dis(self.seq1[i],self.seq2[j])
        self.accum_dis_mat = np.zeros((self.s1_length,self.s2_length))
        self.accum_dis_mat[0,0] =  dis_mat[0,0]
        for i in range(1,self.s1_length):
                self.accum_dis_mat[i,0] = self.accum_dis_mat[i-1,0]+dis_mat[i,0]
        for j in range(1,self.s2_length):
                self.accum_dis_mat[0,j] = self.accum_dis_mat[0,j-1]+dis_mat[0,j]
        for i in range(1, self.s1_length):
                for j in range(1, self.s2_length):
                    choices = [self.accum_dis_mat[i-1, j-1], self.accum_dis_mat[i, j-1], self.accum_dis_mat[i-1, j]]
                    self.accum_dis_mat[i, j] = min(choices) + dis_mat[i,j]
                    
        
        return self.accum_dis_mat[-1,-1]
        
        
    def calc_DTW(self,seq1=[],seq2=[],calc_dis = None): 
        self.seq1 = np.array(seq1)
        self.seq2 = np.array(seq2)
        
        self.s1_length = len(seq1)
        self.s2_length = len(seq2)        
        if calc_dis is None:
            self.calc_dis = lambda X,Y:np.linalg.norm(np.array(X-Y),2)
        else:
            self.calc_dis = calc_dis
            
        self.distance = 0
        self.path_map = []
        
        self.arg = np.zeros((self.s1_length,self.s2_length))
        dis_mat = np.zeros((self.s1_length,self.s2_length))
        accum_dis_mat = np.zeros((self.s1_length,self.s2_length))


        for i in range(self.s1_length):
            for j in range(self.s2_length):
                dis_mat[i,j]=self.calc_dis(self.seq1[i],self.seq2[j])
        self.accum_dis_mat = np.zeros((self.s1_length,self.s2_length))
        self.accum_dis_mat[0,0] =  dis_mat[0,0]

        for i in range(1,self.s1_length):
                accum_dis_mat[i,0] = accum_dis_mat[i-1,0]+dis_mat[i,0]
                self.arg[i,0] = 2
        for j in range(1,self.s2_length):
                accum_dis_mat[0,j] = accum_dis_mat[0,j-1]+dis_mat[0,j]
                self.arg[0,j] = 1
        for i in range(1, self.s1_length):
                for j in range(1, self.s2_length):
                    choices = [accum_dis_mat[i-1, j-1], accum_dis_mat[i, j-1], accum_dis_mat[i-1, j]]
                    self.arg[i,j] = np.argmin(choices)
                    accum_dis_mat[i, j] = min(choices) + dis_mat[i,j]
                    
        
        X,Y = self.s1_length-1,self.s2_length-1
        self.path_map = [[X,Y]]
        while(X or Y):
            if self.arg[X,Y]==0:
                self.path_map.append([X-1,Y-1])
                X-=1
                Y-=1
                
            elif self.arg[X,Y] == 1:
                self.path_map.append([X,Y-1])
                Y=Y-1
            else:
                self.path_map.append([X-1,Y])
                X=X-1
        self.path_map = self.path_map[::-1]
        self.distance = accum_dis_mat[-1,-1]
                
        
        
        return self.distance
    
    
#    def calc_DTW(self):
#        self.path_map = [[0,0]]
#        self.distance = 0
#    
#        
#        
#        i = j = 1
#        while(i<self.s1_length):
#            dis1 = self.calc_dis(self.seq1[i],self.seq2[j])
#            dis2 = self.calc_dis(self.seq1[i],self.seq2[j-1])
#            dis3 = self.calc_dis(self.seq2[j],self.seq1[i-1])
#            
#            if dis1<=dis2 and dis1 <= dis3:
#                self.path_map.append([i,j])
#                i+=1
#                j+=1
#                self.distance += dis1
#            elif dis2<dis1 and dis2 <= dis3:
#                self.path_map.append([i,j-1])
#                i+=1
#                self.distance += dis2
#                
#            else:
#                self.path_map.append([i-1,j])
#                j+=1
#                self.distance += dis3
#                
#            if j>=self.s2_length:
#                for k in self.seq1[i:]:
#                    self.path_map.append([k,j-1])
#                    self.distance+=self.calc_dis(self.seq1[k],self.seq2[j-1])
#                break
#        else:
#            for k in self.seq2[j:]:
#                self.path_map.append([i-1,k])
#                self.distance+=self.calc_dis(self.seq1[i-1],self.seq2[k])







if __name__ == '__main__':
    lis1 = [1, 1, 3, 3, 2, 4]
    lis2 = [1, 3, 3, 2, 2, 4]
    distance = lambda X,Y:abs(X-Y)
    dtw = DTW()
    dtw.calc_DTW(lis1,lis2,distance)

    
    
    