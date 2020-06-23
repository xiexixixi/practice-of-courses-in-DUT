# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:02:47 2020

@author: Lenovo
"""
import tkinter as tk



import time


window = tk.Tk()
window.title('my window')
window.geometry('1000x500')
#
#var = tk.StringVar()
#
#l = tk.Label(window,text = 'OMG!this is tk',bg='green',
#             font = ('Arial',12),width=15,height=2)
#
#l = tk.Label(window,textvariable = var,bg='green',
#             font = ('Arial',12),width=15,height=2)
#
#l.pack()
#on_hit = False
#def hit():
#    global on_hit
#    if not on_hit:
#        on_hit = True
#        var.set("You hit me")
#    else:
#        on_hit = False
#        var.set('')
    

e =tk.Entry(window,show=None)

e.pack()

def insert():
    global e
    var = e.get()
    t.insert('insert',var)
    
b1 = tk.Button(window,text = 'insert cur position',
              width = 15,height = 2,command=insert)
b1.pack()


def insert_end():
    global e
    var = e.get()
    t.insert('end',var)
b2 = tk.Button(window,text = 'insert end',
              width = 15,height = 2,command=insert_end)

b2.pack()

t = tk.Text(window,height =2)
t.pack()


window2 = tk.Tk()

def new_window():
    window2.mainloop()
but = tk.Button(window,text = 'new window',width = 15,height = 2,command=new_window)
but.pack()



canvas = tk.Canvas(window,height=500,width = 500)
filename = tk.PhotoImage(file = 'picture.png')
image = canvas.create_image(0, 0, anchor=tk.NW, image=filename)
canvas.place(relx=0., rely=0.0, anchor=tk.NW,width=500,height=500)



window.mainloop()


