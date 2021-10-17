# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:28:01 2020

@author: Lenovo
"""

import tkinter as tk
from tkinter import ttk
from code4 import *
import matplotlib.pyplot as plt



def enter():
    global e_1,e_2
    sep = [',',' ']
    seqential = e_1.get()
    init_posi = e_2.get()
    
    for s in sep:
        if s in seqential:
            seqential = [eval(c) for c in seqential.split(s)]
            
            break


    return seqential,int(init_posi)


    


def bSSTF():
    global t,frm2
    seqential,init_pos = enter()
    move_seq,MAL,ARL = SSTF(init_pos,seqential)
    fig, ax = plt.subplots()
    for i in range(len(seqential)-1):
        plt.plot(move_seq[i:i+2],[i,i+1])
        x = move_seq[i+1]
        y = i+1
        string = '%d'%x
        
        plt.text(x+0.15, y-0.15, string, ha='center', va='bottom', fontsize=10.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    fig.savefig('picture.png')
    plt.close('all')
    
    
    

    t.delete('1.0','end')
    t.insert('end','响应请求顺序：')
    t.insert('end',str(move_seq).strip('[').strip(']'))
    t.insert('end','\n移臂总量:%d'%MAL)
    t.insert('end','\n平均寻道长度:%d'%ARL)
    


def bSCAN():
    seqential,init_pos = enter()
    move_seq,MAL,ARL = SCAN(init_pos,seqential)
    
    fig = plt.figure()
    for i in range(len(seqential)-1):
        plt.plot(move_seq[i:i+2],[i,i+1])
        x = move_seq[i+1]
        y = i+1
        string = '%d'%x
        
        plt.text(x+0.15, y-0.15, string, ha='center', va='bottom', fontsize=10.5)
    fig.savefig('picture.png')
    plt.close('all')
    
    t.delete('1.0','end')
    t.insert('end','响应请求顺序：')
    t.insert('end',str(move_seq).strip('[').strip(']'))
    t.insert('end','\n移臂总量:%d'%MAL)
    t.insert('end','\n平均寻道长度:%d'%ARL)



win_exp3 = tk.Tk()

win_exp3.title('实验4：磁盘移臂调度算法实验')
win_exp3.geometry('1100x600')

frm1 = tk.Frame(win_exp3,height = 700,width = 300,relief='sunken')
frm1.pack(side = 'left')

frm11 = tk.Frame(frm1,height = 300,width = 400,borderwidth=1)
frm11.pack(side = 'top')

l_1 = tk.Label(frm11,text = "磁盘请求序列:",
             font = ('宋体',12),width=15,height=1)
l_1.place(relx=0.1, rely=0.1,anchor=tk.NW)

l_2 = tk.Label(frm11,text = "初始位置：",
             font = ('宋体',12),width=15,height=1)
l_2.place(relx=0.1, rely=0.4, anchor=tk.NW)


e_1 =tk.Entry(frm11,show=None)
e_1.place(relx=0.4, rely=0.1, anchor=tk.NW)
e_2 =tk.Entry(frm11,show=None)
e_2.place(relx=0.4, rely=0.4, anchor=tk.NW)



frm12 = tk.Frame(frm1,height = 200,width = 400)
frm12.pack(side = 'bottom')

button_font = ('宋体',25)
buttons_121 = []
buttons_121.append(tk.Button(frm12,text = 'SSTF',font = button_font,width = 5,height = 2,relief='groove',command=bSSTF))
buttons_121.append(tk.Button(frm12,text = 'SCAN',font = button_font, width = 5,height = 2,relief='groove',command=bSCAN))

buttons_121[0].place(relx=0.1, rely=0.1, anchor=tk.NW)
buttons_121[1].place(relx=0.9, rely=0.1, anchor=tk.NE,x=-50)



frm2 = tk.Frame(win_exp3,height = 700,width = 900,borderwidth=1)
frm2.place(relx=0.35, rely=0.0, anchor=tk.NW)


t = tk.Text(frm2,height =2)
t.place(relx=0., rely=0.4, anchor=tk.NW,width=600,height=200)

def show_pic():
    
    pic = tk.Toplevel()
    pic.title('实验4：磁盘移臂示意图')
    pic.geometry('500x300')
    
    img_png = tk.PhotoImage(file = 'picture.png')
    label_img = tk.Label(pic, image = img_png)
    label_img.pack()
    
    pic.mainloop()
    

b_pic = tk.Button(frm2,text = '显示移动过程',font = button_font,width = 5,height = 2,relief='groove',command=show_pic)
b_pic.place(relx=0.0, rely=0.01, anchor=tk.NW,width=600,height=250)


#b_quit = tk.Button(frm12,text = '退出',font = button_font,width = 10,height = 1,relief='groove',command=win_exp3.destroy)
#b_quit.place(relx=0.2, rely=0.7, anchor=tk.NW)
win_exp3.mainloop()




