# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:51:24 2020

@author: Lenovo
"""

import tkinter as tk
    
from tkinter import ttk
from code3 import *





window = tk.Tk()
window.title('操作系统实验')
window.geometry('300x550')
window.resizable(width=False, height=False)



label = tk.Label(window,text = "实验列表",bg='gray',
             font = ('宋体',30),width=15,height=2)
label.pack()

button_font = ('宋体',25)

def Processor_scheduling():
    import ex2_ui
def Storage_management():
    import ex3_ui
def Diskarm_shifting_scheduling():
    import ex4_ui
def File_management():
    import ex5_ui
    
    
b2 = tk.Button(window,text = '处理器调度',font = button_font,width = 15,height = 2,relief='groove',command=Processor_scheduling)
b2.pack(padx = 20,pady = 10)



b3 = tk.Button(window,text = '存储管理',font = button_font,width = 15,height = 2,relief='groove',command=Storage_management)
b3.pack(padx = 20,pady = 10)




b4 = tk.Button(window,text = '磁盘移臂调度',font =button_font,width = 15,height = 2,relief='groove',command=Diskarm_shifting_scheduling)
b4.pack(padx = 20,pady = 10)



b5 = tk.Button(window,text = '文件管理',font =button_font,width = 15,height = 2,relief='groove',command=File_management)
b5.pack(padx = 20,pady = 10)

window.mainloop()








