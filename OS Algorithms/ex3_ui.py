# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 20:21:24 2020

@author: Lenovo
"""
import tkinter as tk
from tkinter import ttk
from code3 import *




def enter():
    global e_1,e_2
    sep = [',',' ']
    seqential = e_1.get()
    page_num = e_2.get()
    
    for s in sep:
        if s in seqential:
            seqential = [eval(c) for c in seqential.split(s)]
            
            break


    return seqential,int(page_num)

def config_table(page_num):
    
    global frm2,tree_date
    tree_date = ttk.Treeview(frm2)
    
    lis = ['page'+str(i+1) for i in range(page_num)]
    lis.append('page fault')
    lis.append('Pages swapped out')
    tree_date['columns'] = lis

    w = 80
    for c in tree_date['columns']:
        tree_date.column(c,width=w)
        tree_date.heading(c,text=c)



def show_table(Sm):
    global t
    t.delete('1.0','end')
    for i in range(len(Sm.page_seq)-1,-1,-1):
        txt = '引用页面 %d'%Sm.page_seq[i]
        val = list(Sm.page_table[i].copy())
        val.append(Sm.page_faults[i])
        val.append(Sm.page_out[i])
        val = tuple(val)
        tree_date.insert('',0,text=txt,values=val)
    t.insert('insert',Sm.info_text())
    
    
args = {'relx':0.05, 'rely':0.10, 'anchor':tk.NW,'width':500,'height':200}

def bLRU():
    
    
    seqential,page_num = enter()
    
    
    config_table(page_num)
    tree_date.place(args)
    Sm = Storage_manag(seqential,page_num)
    Sm.LRU()
    info = Sm.info_text()
    
    show_table(Sm)

def bFIFO():

    
    seqential,page_num = enter()
    
    config_table(page_num)
    tree_date.place(args)
    Sm = Storage_manag(seqential,page_num)
    Sm.FIFO()
    info = Sm.info_text()
    
    show_table(Sm)



win_exp3 = tk.Tk()

win_exp3.title('实验3：存储管理')
win_exp3.geometry('900x500')

frm1 = tk.Frame(win_exp3,height = 700,width = 300,relief='sunken')
frm1.pack(side = 'left')

frm11 = tk.Frame(frm1,height = 300,width = 400,borderwidth=1)
frm11.pack(side = 'top')

l_1 = tk.Label(frm11,text = "页面引用序列：",
             font = ('宋体',12),width=15,height=1)
l_1.place(relx=0.1, rely=0.4,anchor=tk.NW)

l_2 = tk.Label(frm11,text = "页面数目：",
             font = ('宋体',12),width=15,height=1)
l_2.place(relx=0.1, rely=0.6, anchor=tk.NW)


e_1 =tk.Entry(frm11,show=None)
e_1.place(relx=0.4, rely=0.4, anchor=tk.NW)
e_2 =tk.Entry(frm11,show=None)
e_2.place(relx=0.4, rely=0.6, anchor=tk.NW)



frm12 = tk.Frame(frm1,height = 200,width = 400)
frm12.pack(side = 'bottom')

button_font = ('宋体',25)
buttons_121 = []
buttons_121.append(tk.Button(frm12,text = 'LRU',font = button_font,width = 7,height = 2,relief='groove',command=bLRU))
buttons_121.append(tk.Button(frm12,text = 'FIFO',font = button_font, width = 7,height = 2,relief='groove',command=bFIFO))

buttons_121[0].place(relx=0.1, rely=0.1, anchor=tk.NW)
buttons_121[1].place(relx=0.9, rely=0.1, anchor=tk.NE,x=-50)



frm2 = tk.Frame(win_exp3,height = 700,width = 900,borderwidth=1)
frm2.place(relx=0.35, rely=0.0, anchor=tk.NW)
tree_date = ttk.Treeview(frm2)
tree_date.place(args)

t = tk.Text(frm2,height =2)
t.place(relx=0.05, rely=0.4, anchor=tk.NW,width=500,height=150)


#b_quit = tk.Button(frm12,text = '退出',font = button_font,width = 10,height = 1,relief='groove',command=win_exp3.destroy)
#b_quit.place(relx=0.2, rely=0.7, anchor=tk.NW)
win_exp3.mainloop()














