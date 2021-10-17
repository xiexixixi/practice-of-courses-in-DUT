# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 02:22:04 2020

@author: Lenovo
"""
import random
import tkinter as tk
from tkinter import ttk
from code5 import *




def enter():
    global e_1,e_2
    name=''
    size = 0
    name = e_1.get()
    size = e_2.get()
    return name,int(size)

def config_table():
    
    global frm2,tree_date
    tree_date = ttk.Treeview(frm2)
    
    tree_date['columns'] = ['address','blocks']

    w = 80
    for c in tree_date['columns']:
        tree_date.column(c,width=w)
        tree_date.heading(c,text=c)



def show_table(dm,text):
    global t
    t.delete('1.0','end')
    
    x=tree_date.get_children()
    for item in x:
        tree_date.delete(item)
    for key,val in dm.vacancy_table.items():
        txt = '序号%d'%key
        val = tuple(val)
        tree_date.insert('',0,text=txt,values=val)
        
    t.insert('end',text)
    
    

def bCreate():
    global dm
    name,size = enter()
    config_table()
    tree_date.place(relx=0., rely=0.05, anchor=tk.NW,width=500,height=200)
    dm.Create_file(name,size)
    
    info = dm.info_text()
    
    show_table(dm,info)

def bDelete():
    
    global dm
    name,size = enter()
    config_table()
    tree_date.place(relx=0., rely=0.05, anchor=tk.NW,width=500,height=200)
    dm.Delete_file(name)
    
    info = dm.info_text()
    show_table(dm,info)

def bGenerate():
    
    global dm
    config_table()
    tree_date.place(relx=0., rely=0.05, anchor=tk.NW,width=500,height=200)
    #随机生成50个文件
    for i in range(50):
        name = str(i)+'.txt'
        size = random.randint(2000,10000)
        dm.Create_file(name,size)

    info = dm.info_text()
    show_table(dm,info)    
    
def bDelall():
    global dm
    config_table()
    tree_date.place(relx=0., rely=0.05, anchor=tk.NW,width=500,height=200)
    
    for i in range(1,50,2):
        name = str(i)+'.txt'
        dm.Delete_file(name)
        
    info = dm.info_text()
    show_table(dm,info)
       
def bSearch():
    global dm
    name,size = enter()
    config_table()
    tree_date.place(relx=0., rely=0.05, anchor=tk.NW,width=500,height=200)
    file = dm.Search_file(name)
    
    if file is None:
        info = 'No such a file'
    else:
        info = file.info_text()
    show_table(dm,info)



dm = disk_manager()

win_exp3 = tk.Tk()

win_exp3.title('实验5：文件管理')
win_exp3.geometry('1000x600')

frm1 = tk.Frame(win_exp3,height = 700,width = 300,relief='sunken')
frm1.pack(side = 'left')

frm11 = tk.Frame(frm1,height = 300,width = 400,borderwidth=1)
frm11.pack(side = 'top')

l_1 = tk.Label(frm11,text = "文件名：",
             font = ('宋体',12),width=15,height=1)
l_1.place(relx=0.1, rely=0.3,anchor=tk.NW)

l_2 = tk.Label(frm11,text = "文件大小：",
             font = ('宋体',12),width=15,height=1)
l_2.place(relx=0.1, rely=0.6, anchor=tk.NW)


e_1 =tk.Entry(frm11,show=None)
e_1.place(relx=0.4, rely=0.3, anchor=tk.NW)
e_2 =tk.Entry(frm11,show=None)
e_2.place(relx=0.4, rely=0.6, anchor=tk.NW)



frm12 = tk.Frame(frm1,height = 300,width = 400)
frm12.pack(side = 'bottom')

button_font = ('宋体',20)
buttons_121 = []
buttons_121.append(tk.Button(frm12,text = '创建',font = button_font,width = 5,height = 2,relief='groove',command=bCreate))
buttons_121.append(tk.Button(frm12,text = '删除',font = button_font, width = 5,height = 2,relief='groove',command=bDelete))
buttons_121.append(tk.Button(frm12,text = '随机生成文件',font = button_font, width = 19,height = 2,relief='groove',command=bGenerate))
buttons_121.append(tk.Button(frm12,text = '删除奇数文件',font = button_font, width = 19,height = 2,relief='groove',command=bDelall))
buttons_121.append(tk.Button(frm12,text = '查找',font = button_font, width = 5,height = 2,relief='groove',command=bSearch))

buttons_121[0].place(relx=0.1, rely=0.0, anchor=tk.NW)
buttons_121[1].place(relx=0.35, rely=0.0, anchor=tk.NW)
buttons_121[2].place(relx=0.1, rely=0.3, anchor=tk.NW)
buttons_121[3].place(relx=0.1, rely=0.6, anchor=tk.NW)
buttons_121[4].place(relx=0.6, rely=0.0, anchor=tk.NW)

frm2 = tk.Frame(win_exp3,height = 700,width = 700,borderwidth=1)
frm2.place(relx=0.35, rely=0.0, anchor=tk.NW)
tree_date = ttk.Treeview(frm2)
tree_date.place(relx=0., rely=0.05, anchor=tk.NW,width=500,height=200)

t = tk.Text(frm2,height =2)
t.place(relx=0., rely=0.42, anchor=tk.NW,width=500,height=250)


#b_quit = tk.Button(frm12,text = '退出',font = button_font,width = 10,height = 1,relief='groove',command=win_exp3.destroy)
#b_quit.place(relx=0.2, rely=0.7, anchor=tk.NW)
win_exp3.mainloop()


