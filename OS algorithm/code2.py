# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:34:49 2020

@author: Lenovo
"""

#处理机调度

import pandas as pd

#数据结构为DataFrame
def DFrame_process(arrival_time,service_time,process_name=None):
    process = pd.DataFrame({'arrival_time':pd.Series(arrival_time),
                'service_time':pd.Series(service_time),
                'run_start_time':0,
                'run_finish_time':0,
                'turnaround_time':0,
                'turnaround_time_with_weight':0})
    process.index = process_name
    return process

def FCFS(process_):
    process = process_.copy()
    process_num = len(process)
    process.sort_values(by=['arrival_time','service_time'],inplace=True)
    
    ready_que = []
    finish_que = []
    time_dic = {}
    
    time = 0
    processor_on = False
    
    p_running = None
    while(len(finish_que)<process_num):
        #判断是否有进程进入就绪队列,并更新就绪队列
        arrival_p = process[process['arrival_time']==time].index
        if len(arrival_p):
            ready_que.extend(arrival_p)
            
            time_dic[time] = ready_que.copy()
            
         #判断是否有进程要结束
        if processor_on and time == process.loc[p_running]['run_finish_time']:
            
            processor_on = False
            finish_que.extend(p_running)
            print("time:%d %s finish"%(time,p_running))
            
            
        #判断是否有进程要运行，更新就绪队列
        
        if not processor_on and len(ready_que):
            
            
            p_running = ready_que[0]
            print("time:%d %s start"%(time,p_running))
            processor_on = True
            process.loc[p_running]['run_start_time'] = time
            process.loc[p_running]['run_finish_time'] = time+process.loc[p_running]['service_time']
            del ready_que[0]
            
            time_dic[time] = ready_que.copy()
    
        time+=1
        #计算周转时间
    process['turnaround_time'] = process['run_finish_time']-process['arrival_time']
    process['turnaround_time_with_weight'] = process['turnaround_time']/process['service_time']

    return process,time_dic


def SJF(process_):
    process = process_.copy()

    process_num = len(process)
    process.sort_values(by=['arrival_time','service_time'],inplace=True)
    
    ready_que = []
    finish_que = []
    time_dic = {}
    
    time = 0
    processor_on = False
    
    p_running = None
    while(len(finish_que)<process_num):
        #判断是否有进程进入就绪队列,并更新就绪队列
        arrival_p = process[process['arrival_time']==time].index
        if len(arrival_p):
            ready_que.extend(arrival_p)
            
            time_dic[time] = ready_que.copy()
            
         #判断是否有进程要结束
        if processor_on and time == process.loc[p_running]['run_finish_time']:
            
            processor_on = False
            finish_que.extend(p_running)
            print("time:%d %s finish"%(time,p_running))
            
            
        #判断是否有进程要运行，更新就绪队列
        
        if not processor_on and len(ready_que):
            p_running = process['service_time'][ready_que].idxmin()
            print("time:%d %s start"%(time,p_running))
            processor_on = True
            process.loc[p_running]['run_start_time'] = time
            process.loc[p_running]['run_finish_time'] = time+process.loc[p_running]['service_time']
            ready_que.remove(p_running)
            time_dic[time] = ready_que.copy()
    
        time+=1
        #计算周转时间
    process['turnaround_time'] = process['run_finish_time']-process['arrival_time']
    process['turnaround_time_with_weight'] = process['turnaround_time']/process['service_time']
    
    return process,time_dic



def RR(process_,q = 1):
    process = process_.copy()
    process_num = len(process)
    process.sort_values(by=['arrival_time','service_time'],inplace=True)
    
    ready_que = []
    remain_time = []
    
    finish_que = []
    time_dic = {}
    
    time = 0
    processor_on = False
    p_running = None
    
    
    
    
    while(len(finish_que)<process_num):
        #判断是否有进程进入就绪队列,并更新就绪队列
        arrival_p = process[process['arrival_time']==time].index
        need_time = list(process['service_time'][arrival_p])
        if len(arrival_p):
            ready_que.extend(arrival_p)
            remain_time.extend(need_time)
            
            time_dic[time] = ready_que.copy()
        
        #取就绪队列队首进程，运行q秒
        if not processor_on and len(ready_que):
            p_running = ready_que[0]
            if process['service_time'][p_running] == remain_time[0]:
                process['run_start_time'][p_running] = time
                
            print("time:%d %s start"%(time,p_running))
            processor_on = True
            remain_time[0]-=1
            
            time_dic[time] = ready_que.copy()
            
            del ready_que[0]
            if not remain_time[0]:
                
                finish_que.extend(p_running)
                print("time:%d %s finish"%(time,p_running))
                process.loc[p_running]['run_finish_time'] = time+1
            else:
                ready_que.append(p_running)
                remain_time.append(remain_time[0])
                time_dic[time] = ready_que.copy()
                
                
            processor_on = False
    
            del remain_time[0]
            
        time+=1
#    process['run_start_time'] = process['arrival_time']+1

    process['turnaround_time'] = process['run_finish_time']-process['arrival_time']
    process['turnaround_time_with_weight'] = process['turnaround_time']/process['service_time']
          
    return process,time_dic

def HRN(process_):
    process = process_.copy()

    process_num = len(process)
    process.sort_values(by=['arrival_time','service_time'],inplace=True)
    
    ready_que = []
    finish_que = []
    time_dic = {}
    
    time = 0
    processor_on = False
    
    p_running = None
    while(len(finish_que)<process_num):
        #判断是否有进程进入就绪队列,并更新就绪队列
        arrival_p = process[process['arrival_time']==time].index
        if len(arrival_p):
            ready_que.extend(arrival_p)
            time_dic[time] = ready_que.copy()
            
         #判断是否有进程要结束
        if processor_on and time == process.loc[p_running]['run_finish_time']:
            
            processor_on = False
            finish_que.extend(p_running)
            print("time:%d %s finish"%(time,p_running))
            
            
        #判断是否有进程要运行，更新就绪队列
        
        if not processor_on and len(ready_que):
            
            R = time - process['arrival_time'][ready_que]/process['service_time'][ready_que]
            p_running = R.idxmax()
            print("time:%d %s start"%(time,p_running))
            processor_on = True
            process.loc[p_running]['run_start_time'] = time
            process.loc[p_running]['run_finish_time'] = time+process.loc[p_running]['service_time']
            ready_que.remove(p_running)
            time_dic[time] = ready_que.copy()
    
        time+=1
        #计算周转时间
    process['turnaround_time'] = process['run_finish_time']-process['arrival_time']
    process['turnaround_time_with_weight'] = process['turnaround_time']/process['service_time']
    
    return process,time_dic


if __name__ == '__main__':

    arrival_time=[4,2,0,6,8]
    service_time=[4,6,3,5,2]
    process_name=['A','B','C','D','E']
    
    process = DFrame_process(arrival_time,service_time,process_name)
    process,time_dic = HRN(process)

    

















