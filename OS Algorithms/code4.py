# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:47:31 2020

@author: Lenovo
"""

import numpy as np


seq = [90,58,55,39,38,18,150,160,184]
init_pos = 100

def SSTF(init_pos,seq):
    move_seq = [init_pos]
    pos = init_pos
    DR_seq = sorted(seq)
    
    
    while(len(DR_seq)):
        
        dis = [abs(ad - pos) for ad in DR_seq]
        next_move = np.argmin(dis)
        move_seq.append(DR_seq[next_move])
        pos = DR_seq[next_move]
        del DR_seq[next_move]
    
    MAL = sum([abs(move_seq[i+1]-move_seq[i]) for i in range(len(seq))])
    ARL = MAL/len(seq)
    
    #顺序，移臂总量，平均寻道长度
    return move_seq,MAL,ARL


def SCAN(init_pos,seq,inc_derection = True):
    
    move_seq = []
    
    DR_seq = seq.copy()
    DR_seq.append(init_pos)
    DR_seq = sorted(DR_seq)
    
    b = [_==init_pos for _ in DR_seq]
    if inc_derection:
        
        idx = np.argmax(b)
        move_seq.extend(DR_seq[idx:].copy())
        move_seq.extend(DR_seq[idx-1::-1].copy())
        
    else:
        idx = np.argmax(b)+sum(b)
        move_seq.extend(DR_seq[idx-1::-1])
        move_seq.extend(DR_seq[idx:])
    MAL = sum([abs(move_seq[i+1]-move_seq[i]) for i in range(len(move_seq)-1)])
    ARL = MAL/(len(move_seq)-1)
    
    return move_seq,MAL,ARL
    
    
if __name__ == '__main__':
    move_seq,MAL,ARL = SCAN(init_pos,seq)
















