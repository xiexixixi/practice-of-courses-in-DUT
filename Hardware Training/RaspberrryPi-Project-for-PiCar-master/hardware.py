import serial
import time
import RPi.GPIO as GPIO

def num2char(num):
    if num < 10:
        return '00'+str(num)
    elif num<100:
        return '0'+str(num)
    else:
        return str(num)

class ControlHardware:
    def __init__(self):
        # open serial
        self.ser=serial.Serial('/dev/ttyAMA0',9600)
        while(not self.ser.isOpen()):
            print("serial not opened")
    def action(self, instruction):
        self.ser.write(instruction)
        time.sleep(0.1)
    def stop(self):

        instruction = '#CP@@@@@@'
        self.action(instruction)
    def font(self):
 
        instruction = '#CF@@@@@@'
        self.action(instruction)
    def backward(self):
 
        instruction = '#CB@@@@@@'
        self.action(instruction)
    def left(self):

        instruction = '#CL@@@@@@'
        self.action(instruction)
    def right(self):

        instruction = '#CR@@@@@@'
        self.action(instruction)
    def leftg(self):

        instruction = '#Cl@@@@@@'
        self.action(instruction)
    def rightg(self):

        instruction = '#Cr@@@@@@'
        self.action(instruction)
    def left_angle(self,angle):
        
        instruction = '#Ca'+num2char(angle)+'@@@'
        print instruction
        self.action(instruction)
    def right_angle(self,angle):

        instruction = '#Cc'+num2char(angle)+'@@@'
        self.action(instruction)
    def left_right_angle(self,langle,rangle):

        instruction = '#Cb'+num2char(langle)+num2char(rangle)
        self.action(instruction)
    def speedUp(self):

        instruction = '$CA@@@@@@'
        self.action(instruction)
    def speedDown(self):

        instruction = '$CP@@@@@@'
        self.action(instruction)




if __name__ == "__main__":
    h = ControlHardware()

   
    
    
    #h.font()
    #print 1
    
    #time.sleep(2)
    
    #h.speedUp()
    #time.sleep(3)
    #h.backward()
    #print 2
    #time.sleep(2)
    #h.stop()
    #h.speedDown()
    #time.sleep(3)
    #h.right()
    #time.sleep(2)
    #h.stop()
    #time.sleep(1)
    #h.left()
    #print 3
    #h.stop()
    #time.sleep(3)
    
    #h.right()
    #print 4
    #time.sleep(3)
    #h.stop()
    #h.stop()
    #time.sleep(3)
    
    #h.leftg()
    #print 5
    #h.stop()
    #time.sleep(3)
    
    #h.rightg()
    #print 6
    
    #h.stop()
    #time.sleep(3)
    #time.sleep(3)
    #h.left_angle(45)
    #time.sleep(3)
    #h.right_angle(150)
    #time.sleep(3)
    #
    h.right_angle(70)








