# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:52:26 2020

@author: Lenovo
"""
import os
import socket
import speak
import time
from pygame import mixer
host = socket.gethostname()
print(host)

host = 'DESKTOP-GPH3M5L'
port = 8000

msg = 'recog'

disease = ['肺癌','红斑狼疮','糖尿病','艾滋病','阿尔兹海默症','肝癌','痛风','智障']

meals = ['苹果','香蕉','橘子','营养液','花椒','大料']

try:
    client = socket.socket()  # 声明socket类型，同时生成socket连接对象
    client.connect((host, port))  # 链接服务器的ip + 端口

    print('connected!')
    client.send(msg.encode("utf-8"))


except:
    print('connect failed')
    pass



while True:

    mes = str(client.recv(1024))

    if mes == 'finish recording!':
        # TODO 说话，录制完毕
        speak.Speak('录制完毕')
        mixer.init()
        mixer.music.load('speak.mp3')
        mixer.music.play()
        print('finished recording')
        break

    elif mes[:8] == 'recogfin':
        #TODO 说出是谁得了啥病

        speak.Speak('{0}号病人，身患{1}，您今天可以吃：{2}'.format(mes[9:],disease[eval(mes[9:])],meals[eval(mes[9:])]))
        os.system('mplayer speak.mp3')
        print('{0}号病人，身患{1}，已经没救了'.format(mes[9:],disease[eval(mes[9:])]))
        break











