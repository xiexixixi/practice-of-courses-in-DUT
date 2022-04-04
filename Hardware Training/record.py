# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:52:26 2020

@author: Lenovo
"""
import os

import socket
from pygame import mixer
import speak


host = socket.gethostname()
port = 8000
host = 'DESKTOP-GPH3M5L'
msg = 'record'
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
        os.system('mplayer speak.mp3')
        print('finished recording')
        break

    elif mes[:8] == 'recogfin':
        #TODO 说出是谁得了啥病
        print(mes[9:])
        break











