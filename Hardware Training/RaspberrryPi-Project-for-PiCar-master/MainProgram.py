#! /usr/bin/python
import socket
import os
from enum import IntEnum
#from MotorModule.Motor import Motor
import traceback
import threading
import time
import hardware as hard
#from SteeringModule.Steering import Steering
#import cv2
import numpy
#from OledModule.OLED import OLED
h = hard.ControlHardware()
def getLocalIp():
    '''Get the local ip'''
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('8.8.8.8',80))
        ip=s.getsockname()[0]
    finally:
        s.close()
    return ip

'''def cameraAction(command):
    if command=='Record':
        print("Record")

    elif command=='Recognize':
        print("Recognize")

    elif command=='Revive':
        print("Revive")'''



#def motorAction('''motor,command):
def motorAction(command):
    '''Set the action of motor according to the command'''
    if command=='DirForward':
        print("ahead")
        h.font()

        # motor.ahead()
    elif command=='DirBack':
        h.backward()
        print("rear")
        # motor.rear()
    elif command=='DirLeft':
        h.left()
        print("left")
        # motor.left()
    elif command=='DirRight':
        h.right()
        print("right")
        #motor.right()
    elif command=='DirStop':
        h.stop()
        print("stop")
        # motor.stop()
    elif command=='C':
        h.rightg()
        print("C")
    elif command=='AC':
        h.leftg()
        print("AC")
    elif command=='Record':
        os.chdir("/home/pi/New")
        os.system('python record.py')
        print("Record")

    elif command=='Recognize':
        os.chdir("/home/pi/New")
        os.system('python recog1.py')
        print("Recognize")

    elif command=='Revive':
        h.left_right_angle(45,45)
        time.sleep(3)
        h.left_right_angle(135,135)
        time.sleep(3)
        h.left_right_angle(45,45)
        time.sleep(3)
        h.left_right_angle(135,135)
        time.sleep(3)
        h.left_right_angle(90,90)
        print("Revive")

def setCameraAction(command):
    if command=='Record' or command=='Recognize' or command=='Revive':
        return command
    else:
        return 'CamStop'



def main():
    '''The main thread, control the motor'''
    host=getLocalIp()
    print('localhost ip :'+host)
    port=5050

    #Init the tcp socket
    tcpServer=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    tcpServer.bind((host,port))
    tcpServer.setblocking(0) #Set unblock mode
    tcpServer.listen(5)

    '''#Init motor module
    motor=Motor(5,21,22,23,24,13)
    motor.setup()

    #Init steering module
    steer=Steering(14,0,180,15,90,180,36,160)
    steer.setup()
    global cameraActionState #Set a state variable for steering module
    cameraActionState='CamStop'

    #Init oled module
    oled=OLED(16,20,0,0)
    oled.setup()

    oled.writeArea1(host)
    oled.writeArea3('State:')
    oled.writeArea4(' Disconnect') '''

    while True:
        try:
            time.sleep(0.001)
            (client,addr)=tcpServer.accept()
            print('accept the client!')
            #oled.writeArea4(' Connect')
            client.setblocking(0)
            while True:
                time.sleep(0.001)
                #cameraAction(cameraActionState)
                try:
                    data=client.recv(1024)
                    data=bytes.decode(data)
                    if(len(data)==0):
                        print('client is closed')
                        #oled.writeArea4(' Disconnect')
                        break
                    #motorAction(motor,data)
                    motorAction(data)
                    #cameraActionState=setCameraAction(data)
                except socket.error:
                    continue
                except KeyboardInterrupt as e:
                    raise e
        except socket.error:
            pass
        except KeyboardInterrupt:
            #motor.clear()
            #steer.cleanup()
            tcpServer.close()
            #oled.clear()
        except Exception as e1:
            traceback.print_exc()
            #motor.clear()
            #steer.cleanup()
            tcpServer.close()
            #oled.clear()
main()
