# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 00:00:39 2020

@author: Lenovo
"""

import socket
import cv2
import os
import numpy as np
from PIL import Image


# server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

CascadeClassifier_path = r'C:/Users/Lenovo/AppData/Local/Programs/Python/Python374/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
# CascadeClassifier_path = r'haarcascade_frontalface_default.xml'

image_path = r'C:/Users/Lenovo/Desktop/hello'
xml_saved_path = r'E:\face'

camera_device = 'http://192.168.43.205:8080/?action=stream'

def generate_data(image_path,sample_num=10):

    cap = cv2.VideoCapture(camera_device)
    face_detector = cv2.CascadeClassifier(CascadeClassifier_path)
    face_id = input('\n enter user id:')

    print('\n Initializing face capture. Look at the camera and wait ...')
    count = 0

    while True:

        # 从摄像头读取图片

        sucess, img = cap.read()

        # 转为灰度图片

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测人脸

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))
            count += 1
            print('录入%d张'%count)

            # 保存图像
            cv2.imwrite(image_path + "/User." + str(face_id) + '.' + str(count) + '.jpg', gray[y: y + h, x: x + w])
            cv2.imshow('image', img)

        k = cv2.waitKey(1)

        if k == 27:   # 通过esc键退出摄像
            break

        elif count >= sample_num:  # 得到1000个样本后退出摄像
            break

    # 关闭摄像头
    cap.release()
    cv2.destroyAllWindows()

def getImagesAndLabels(path,detector):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # join函数的作用？
    faceSamples = []
    ids = []
    print(imagePaths)
    for imagePath in imagePaths:
        print(imagePath)
        PIL_img = Image.open(imagePath).convert('L')   # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id)
    return faceSamples, ids

def train(image_path,xml_saved_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(CascadeClassifier_path)


    print('Training faces. It will take a few seconds. Wait ...')

    faces, ids = getImagesAndLabels(image_path,detector)
    recognizer.train(faces, np.array(ids))

    recognizer.write(xml_saved_path + r'\trainer.yml')
    print("{0} faces trained. Exiting Program".format(len(np.unique(ids))))

def reco(xml_saved_path):

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(xml_saved_path+'/trainer.yml')
    faceCascade = cv2.CascadeClassifier(CascadeClassifier_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    idnum = 0

    names = [0,1,2,3,4,5,6,7,8,9,10]

    cam = cv2.VideoCapture(camera_device)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    co = 0
    ids = []
    while True:
        if co<15:

            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH))
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                ids.append(idnum)
                co+=1
                print(co)
                if confidence < 100:
                    idnum = names[idnum]
                    confidence = "{0}%".format(round(100 - confidence))
                else:
                    idnum = "unknown"
                    confidence = "{0}%".format(round(100 - confidence))

                cv2.putText(img, str(idnum), (x+5, y-5), font, 1, (0, 0, 255), 1)
                cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (0, 0, 0), 1)

            cv2.imshow('camera', img)
            k = cv2.waitKey(10)
            if k == 27:
                break
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
    print(ids)
    tu=sorted([(np.sum(np.array(ids)==i),i) for i in set(ids)])[-1][1]

    return tu



host = socket.gethostname()
port = 8000
server = socket.socket()  # 1.声明协议类型，同时生成socket链接对象
server.bind((host, port))  # 绑定要监听端口=(服务器的ip地址+任意一个端口)
server.listen(5)  # 监听

print("waiting")

while True:
	conn, addr = server.accept()
	print("收到来自{}请求\n".format(addr))
	while True:
		data = conn.recv(1024)
		if not data:
			print("client has lost...")
			break
		elif data.decode('utf-8') == 'record':
    		#TODO,speak '正在录入信息'

			generate_data(image_path)
			print('finished recording')
			conn.send(bytes('finish recording!',encoding='utf-8'))
			break

		elif data.decode('utf-8') == 'recog':
			train(image_path,xml_saved_path)
			name_id = reco(xml_saved_path)
            #TODO,speak '识别完毕'，什么病
			conn.send(bytes('recogfin '+str(name_id),encoding='utf-8'))
			break



