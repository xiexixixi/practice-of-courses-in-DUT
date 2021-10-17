# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:21:23 2020

@author: Lenovo
"""

import numpy as np



class Storage_manag():
    def __init__(self,seq,frame_num):
        self.frame_num = frame_num
        self.page_seq = np.array(seq)
        self.page_faults = np.zeros_like(self.page_seq)
        self.page_out = np.zeros_like(self.page_seq)
        self.page_table = np.zeros((len(self.page_seq),self.frame_num),dtype=np.int32)
        self.page_faults[0] = 1
        self.page_table[0][0] = self.page_seq[0]
        
        self.page_fault_num = None
        self.page_fault_ratio = None
    def FIFO(self):
        count = 0
        for i in range(1,len(self.page_seq)):
            
            page = self.page_seq[i]
            self.page_table[i] = self.page_table[i-1].copy()
            
            if page not in self.page_table[i]:
                if (self.page_table[i]==0).any():
                    self.page_table[i][self.page_table[i].argmin()] = page
                    self.page_faults[i] = 1
                    continue
                
                
                self.page_faults[i] = 1
                self.page_out[i] = self.page_table[i][count]
                self.page_table[i][count]= page
                count = (count+1)%self.frame_num

        #statistics
        
        self.page_fault_num = sum(self.page_faults)
        self.page_fault_ratio = self.page_fault_num/len(self.page_seq) 
        
    
    def LRU(self):

        for i in range(1,len(self.page_seq)):
            page = self.page_seq[i]
            self.page_table[i] = self.page_table[i-1].copy()
            
            if page not in self.page_table[i]:
                if (self.page_table[i]==0).any():
                    self.page_table[i][self.page_table[i].argmin()] = page
                    self.page_faults[i] = 1
                    continue
                
                self.page_faults[i] = 1
                self.page_out[i] = self.page_table[i][0]
                self.page_table[i][0] = page
                lis = list(self.page_table[i])
                lis.append(lis[0])
                del lis[0]
                self.page_table[i] = np.array(lis,dtype=np.int32)
                
            else:
                idx = np.where(self.page_table[i] == page)[0][0]
                lis = list(self.page_table[i])
                lis.append(lis[idx])
                del lis[idx]
                self.page_table[i] = np.array(lis,dtype=np.int32)
                
        
        
        #statistics
        
        self.page_fault_num = sum(self.page_faults)
        self.page_fault_ratio = self.page_fault_num/len(self.page_seq)        
    
    
        
    def show_info(self):
        
        print("页面数目：",self.frame_num)
        print("页面引用序列：",self.page_seq)
        print("淘汰页面序列：",self.page_out)
        print("缺页次数：",self.page_fault_num)
        print("缺页率：",self.page_fault_ratio)
        
    def info_text(self):
        info = "页面数目：{}\n页面引用序列：{}\n淘汰页面序列：{}\n缺页次数：{}\n缺页率：{}\n".format(
                self.frame_num,self.page_seq,self.page_out,self.page_fault_num,self.page_fault_ratio)

        return info
if __name__ == '__main__':
    seq = [4,3,2,1,4,3,5,4,3,2,1,5]
    frame_num = 3
    #a = Storage_manag(seq,3)
    #a.FIFO()
    #a.show_info()
    
    
    a = Storage_manag(seq,3)
    a.LRU()
    a.show_info()
    

















