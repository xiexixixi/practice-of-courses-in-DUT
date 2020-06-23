# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:44:00 2020

@author: Lenovo
"""
import tkinter as tk
from tkinter import ttk
from code2 import *





def enter():
    global e_1,e_2,e_3
    sep = [',',' ']
    arrival_time = e_1.get()
    service_time = e_2.get()
    process_name = e_3.get()
    
    for s in sep:
        if s in arrival_time:
            arrival_time = [eval(c) for c in arrival_time.split(s)]
        
            break
    
    for s in sep:
        if s in service_time:
            service_time = [eval(c) for c in service_time.split(s)] 
            break
    for s in sep:
        if s in process_name:
            process_name = process_name.split(s)
            break
        
    return arrival_time,service_time,process_name

def config_table():
    
    global frm2,tree_date
    tree_date = ttk.Treeview(frm2)
    tree_date['columns'] = ['到达时间',
                             '服务时间',
                             '开始运行时间',
                             '运行结束时间',
                             '周转时间',
                             '带权周转时间']
    w = 80
    tree_date.column('到达时间',width=w)
    tree_date.column('服务时间',width=w)
    tree_date.column('开始运行时间',width=w)
    tree_date.column('运行结束时间',width=w)
    tree_date.column('周转时间',width=w)
    tree_date.column('带权周转时间',width=w)
    
    
    tree_date.heading('到达时间',text='到达时间')
    tree_date.heading('服务时间',text='服务时间')
    tree_date.heading('开始运行时间',text='开始运行时间')
    tree_date.heading('运行结束时间',text='运行结束时间')
    tree_date.heading('周转时间',text='周转时间')
    tree_date.heading('带权周转时间',text='带权周转时间')


def show_result(process):
    for i in range(len(process)-1,-1,-1):
        txt = process.index[i]
        val = tuple(process.iloc[i])
        tree_date.insert('',0,text=txt,values=val)
    
    
    


def bFCFS():

    
    arrival_time,service_time,process_name = enter()
    config_table()
    tree_date.place(relx=0., rely=0.1, anchor=tk.NW,width=600,height=300)
    process = DFrame_process(arrival_time,service_time,process_name)
    process,time_dic = FCFS(process)
    show_result(process)


def bRR():
    arrival_time,service_time,process_name = enter()
    config_table()
    
    tree_date.place(relx=0., rely=0.1, anchor=tk.NW,width=600,height=300)
    process = DFrame_process(arrival_time,service_time,process_name)
    process,time_dic = RR(process)
    show_result(process)

def bSJF():
    arrival_time,service_time,process_name = enter()
    config_table()
    tree_date.place(relx=0., rely=0.1, anchor=tk.NW,width=600,height=300)
    process = DFrame_process(arrival_time,service_time,process_name)
    process,time_dic = SJF(process)
    show_result(process)

def bHRN():
    arrival_time,service_time,process_name = enter()
    config_table()
    
    tree_date.place(relx=0., rely=0.1, anchor=tk.NW,width=600,height=300)
    process = DFrame_process(arrival_time,service_time,process_name)
    process,time_dic = HRN(process)
    show_result(process)


win_exp2 = tk.Tk()

win_exp2.title('实验2：处理器调度')
win_exp2.geometry('1000x500')

frm1 = tk.Frame(win_exp2,height = 700,width = 300,relief='sunken')
frm1.pack(side = 'left')

frm11 = tk.Frame(frm1,height = 300,width = 400,borderwidth=1)
frm11.pack(side = 'top')

l_1 = tk.Label(frm11,text = "进程到达时间：",
             font = ('宋体',12),width=15,height=1)
l_1.place(relx=0.1, rely=0.3,anchor=tk.NW)

l_2 = tk.Label(frm11,text = "进程服务时间：",
             font = ('宋体',12),width=15,height=1)
l_2.place(relx=0.1, rely=0.5, anchor=tk.NW)

l_3 = tk.Label(frm11,text = "进程名：",
             font = ('宋体',12),width=15,height=1)
l_3.place(relx=0.1, rely=0.7,anchor=tk.NW )

e_1 =tk.Entry(frm11,show=None)
e_1.place(relx=0.4, rely=0.3, anchor=tk.NW)
e_2 =tk.Entry(frm11,show=None)
e_2.place(relx=0.4, rely=0.5, anchor=tk.NW)
e_3 =tk.Entry(frm11,show=None)
e_3.place(relx=0.4, rely=0.7, anchor=tk.NW)


frm12 = tk.Frame(frm1,height = 200,width = 400)
frm12.pack(side = 'bottom')

button_font = ('宋体',25)
buttons_121 = []
buttons_121.append(tk.Button(frm12,text = 'FCFS',font = button_font,width = 7,height = 1,relief='groove',command=bFCFS))
buttons_121.append(tk.Button(frm12,text = 'RR',font = button_font, width = 7,height = 1,relief='groove',command=bRR))
buttons_121.append(tk.Button(frm12,text = 'SJF',font = button_font,width = 7,height = 1,relief='groove',command=bSJF))
buttons_121.append(tk.Button(frm12,text = 'HRN',font = button_font,width = 7,height = 1,relief='groove',command=bHRN))

buttons_121[0].place(relx=0.1, rely=0.1, anchor=tk.NW)
buttons_121[1].place(relx=0.1, rely=0.7, anchor=tk.SW)
buttons_121[2].place(relx=0.9, rely=0.1, anchor=tk.NE,x=-50)
buttons_121[3].place(relx=0.9, rely=0.7, anchor=tk.SE,x=-50)



frm2 = tk.Frame(win_exp2,height = 700,width = 900,borderwidth=1)
frm2.place(relx=0.35, rely=0.0, anchor=tk.NW)

tree_date = ttk.Treeview(frm2)
config_table()
tree_date.place(relx=0., rely=0.1, anchor=tk.NW,width=600,height=300)

win_exp2.mainloop()
'''
2 4 6 8 10 12 14
3 1 3 4 2 1 5
a s d f g w e
'''