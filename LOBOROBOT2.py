'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
@  文件名：LOBOROBOT2.py 
@  驅動智慧小車的基本運動庫函數
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import time
import math
import smbus
from gpiozero import LED

Dir = ['forward', 'backward']

class PCA9685:

    # PCA9685 的暫存器 (register)
    __SUBADR1            = 0x02
    __SUBADR2            = 0x03
    __SUBADR3            = 0x04
    __MODE1              = 0x00
    __PRESCALE           = 0xFE
    __LED0_ON_L          = 0x06
    __LED0_ON_H          = 0x07
    __LED0_OFF_L         = 0x08
    __LED0_OFF_H         = 0x09
    __ALLLED_ON_L        = 0xFA
    __ALLLED_ON_H        = 0xFB
    __ALLLED_OFF_L       = 0xFC
    __ALLLED_OFF_H       = 0xFD
    

    def __init__(self, address, debug=False):
        self.bus = smbus.SMBus(1)
        self.address = address
        self.debug = debug
        if (self.debug):
            print("Reseting PCA9685")
        self.write(self.__MODE1, 0x00)

    def write(self, reg, value):
        "Writes an 8-bit value to the specified register/address"
        self.bus.write_byte_data(self.address, reg, value)  # i2c write, addr=0x40
        if (self.debug):
            print("I2C: Write 0x%02X to register 0x%02X" % (value, reg))

    def read(self, reg):
        "Read an unsigned byte from the I2C device"
        result = self.bus.read_byte_data(self.address, reg) # i2c read, addr=0x40
        if (self.debug):
            print("I2C: Device 0x%02X returned 0x%02X from reg 0x%02X" % (self.address, result & 0xFF, reg))
        return result

    def setPWMFreq(self, freq):
        "Sets the PWM frequency"
        prescaleval = 25000000.0    # 25MHz
        prescaleval /= 4096.0       # 12-bit
        prescaleval /= float(freq)
        prescaleval -= 1.0
        if (self.debug):
            print("Setting PWM frequency to %d Hz" % freq)
            print("Estimated pre-scale: %d" % prescaleval)
        prescale = math.floor(prescaleval + 0.5)
        if (self.debug):
            print("Final pre-scale: %d" % prescale)

        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10        # sleep
        self.write(self.__MODE1, newmode)        # go to sleep
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def setPWM(self, channel, on, off):
        "Sets a single PWM channel"
        self.write(self.__LED0_ON_L + 4*channel, on & 0xFF)
        self.write(self.__LED0_ON_H + 4*channel, on >> 8)
        self.write(self.__LED0_OFF_L + 4*channel, off & 0xFF)
        self.write(self.__LED0_OFF_H + 4*channel, off >> 8)
        if (self.debug):
            print("channel: %d  LED_ON: %d LED_OFF: %d" % (channel,on,off))

    def setDutycycle(self, channel, pulse):
        self.setPWM(channel, 0, int(pulse * (4096 / 100)))

    def setLevel(self, channel, value):
        if (value == 1):
            self.setPWM(channel, 0, 4095)
        else:
            self.setPWM(channel, 0, 0)

# 控制小車行進、後退、轉彎
class LOBOROBOT():
    def __init__(self):
        self.PWMA = 0
        self.AIN1 = 2
        self.AIN2 = 1

        self.PWMB = 5
        self.BIN1 = 3
        self.BIN2 = 4

        self.PWMC = 6
        self.CIN2 = 7
        self.CIN1 = 8

        self.PWMD = 11
        self.DIN1 = 25        # GPIO口
        self.DIN2 = 24        # GPIO口
        
        self.pwm = PCA9685(0x40, debug=False)
        self.pwm.setPWMFreq(50)
        self.motorD1 = LED(self.DIN1)  # 方向口1，設置為輸出模式為LED類型
        self.motorD2 = LED(self.DIN2)  # 方向口2，設置為輸出模式為LED類型

    def MotorRun(self, motor, index, speed):
        if speed > 100:
            return
        if(motor == 0):
            self.pwm.setDutycycle(self.PWMA, speed)
            if(index == Dir[0]):
                self.pwm.setLevel(self.AIN1, 0)
                self.pwm.setLevel(self.AIN2, 1)
            else:
                self.pwm.setLevel(self.AIN1, 1)
                self.pwm.setLevel(self.AIN2, 0)
        elif(motor == 1):
            self.pwm.setDutycycle(self.PWMB, speed)
            if(index == Dir[0]):
                self.pwm.setLevel(self.BIN1, 1)
                self.pwm.setLevel(self.BIN2, 0)
            else:
                self.pwm.setLevel(self.BIN1, 0)
                self.pwm.setLevel(self.BIN2, 1)
        elif(motor == 2):
            self.pwm.setDutycycle(self.PWMC,speed)
            if(index == Dir[0]):
                self.pwm.setLevel(self.CIN1,1)
                self.pwm.setLevel(self.CIN2,0)
            else:
                self.pwm.setLevel(self.CIN1,0)
                self.pwm.setLevel(self.CIN2,1)
        elif(motor == 3):
            self.pwm.setDutycycle(self.PWMD,speed)
            if (index == Dir[0]):
                self.motorD1.off()    # DIn1設置為低電平
                self.motorD2.on()     # DIn2設置為高電平
                #GPIO.output(self.DIN1,0)
                #GPIO.output(self.DIN2,1)
            else:
                self.motorD1.on()    # DIn1設置為高電平
                self.motorD2.off()   # DIn2設置為低電平
                #GPIO.output(self.DIN1,1)
                #GPIO.output(self.DIN2,0)

    def MotorStop(self, motor):
        if (motor == 0):            # 左前輪
            self.pwm.setDutycycle(self.PWMA, 0)
        elif(motor == 1):           # 右前輪
            self.pwm.setDutycycle(self.PWMB, 0)
        elif(motor == 2):           # 左後輪
            self.pwm.setDutycycle(self.PWMC, 0)
        elif(motor == 3):           # 右後輪
            self.pwm.setDutycycle(self.PWMD, 0)
    
    def moveforward_ofs(self, speed, l_ofs, r_ofs, t_time):     # 前進
        self.MotorRun(0,'forward',speed+l_ofs)  # 左前輪
        self.MotorRun(1,'forward',speed+r_ofs)  # 右前輪
        self.MotorRun(2,'forward',speed+l_ofs)  # 左後輪
        self.MotorRun(3,'forward',speed+r_ofs)  # 右後輪
        if t_time > 0: time.sleep(t_time)  # 等幾秒，可以有小數點

    def moveforward(self,speed,t_time):          # 前進
        self.MotorRun(0,'forward',speed)
        self.MotorRun(1,'forward',speed)
        self.MotorRun(2,'forward',speed)
        self.MotorRun(3,'forward',speed)
        if t_time > 0: time.sleep(t_time)  # 等幾秒，可以有小數點

    def movebackward(self,speed,t_time):        # 後退
        self.MotorRun(0,'backward',speed)
        self.MotorRun(1,'backward',speed)
        self.MotorRun(2,'backward',speed)
        self.MotorRun(3,'backward',speed)
        if t_time > 0: time.sleep(t_time)  # 等幾秒，可以有小數點

    def moveLeft(self,speed,t_time):        # 左移
        self.MotorRun(0,'backward',speed)
        self.MotorRun(1,'forward',speed)
        self.MotorRun(2,'forward',speed)
        self.MotorRun(3,'backward',speed)
        if t_time > 0: time.sleep(t_time)  # 等幾秒，可以有小數點

    def moveRight(self,speed,t_time):       # 右移
        self.MotorRun(0,'forward',speed)
        self.MotorRun(1,'backward',speed)
        self.MotorRun(2,'backward',speed)
        self.MotorRun(3,'forward',speed)
        if t_time > 0: time.sleep(t_time)  # 等幾秒，可以有小數點

    def turnLeft(self,speed,t_time):        # 左轉
        self.MotorRun(0,'backward',speed)
        self.MotorRun(1,'forward',speed)
        self.MotorRun(2,'backward',speed)
        self.MotorRun(3,'forward',speed)
        if t_time > 0: time.sleep(t_time)  # 等幾秒，可以有小數點

    def turnRight(self,speed,t_time):       # 右轉
        self.MotorRun(0,'forward',speed)
        self.MotorRun(1,'backward',speed)
        self.MotorRun(2,'forward',speed)
        self.MotorRun(3,'backward',speed)
        if t_time > 0: time.sleep(t_time)  # 等幾秒，可以有小數點

    def backward_Left(self,speed,t_time):   # 後左退
        self.MotorRun(0,'backward',speed)
        self.MotorStop(1)
        self.MotorStop(2)
        self.MotorRun(3,'backward',speed)
        if t_time > 0: time.sleep(t_time)  # 等幾秒，可以有小數點

    def backward_Right(self,speed,t_time):  # 後右退
        self.MotorStop(0)
        self.MotorRun(1,'backward',speed)
        self.MotorRun(2,'backward',speed)
        self.MotorStop(3)
        if t_time > 0: time.sleep(t_time)  # 等幾秒，可以有小數點

    def t_stop(self,t_time):                # 停止
        self.MotorStop(0)
        self.MotorStop(1)
        self.MotorStop(2)
        self.MotorStop(3)
        if t_time > 0: time.sleep(t_time)  # 等幾秒，可以有小數點

    # 設置二為雲台脈衝寬度
    def set_servo_pulse(self,channel,pulse):
        pulse_length = 1000000    # 1,000,000 us per second
        pulse_length //= 60       # 60 Hz
        print('{0}us per period'.format(pulse_length))
        pulse_length //= 4096     # 12 bits of resolution
        print('{0}us per bit'.format(pulse_length))
        pulse *= 1000
        pulse //= pulse_length
        self.pwm.setPWM(channel, 0, pulse)

    # 設置二為雲台角度函數
    def set_servo_angle(self,channel,angle,t_time):
        angle=4096*((angle*11)+500)/20000
        self.pwm.setPWM(channel,0,int(angle))
        if t_time > 0: time.sleep(t_time)  # 等幾秒，可以有小數點

if __name__ == "__main__":
                
    # TEST
    clbrobot = LOBOROBOT()
    
    # 頻率設置為50hz，適用於舵機系統（攝像頭雲台）
    clbrobot.set_servo_angle(10, 70, 1)     # 底座舵機 70   （相機右轉）
    clbrobot.set_servo_angle(10, 110, 1)    # 底座舵機 110  （相機左轉）
    clbrobot.set_servo_angle( 9, 25, 1)     # 頂部舵機 25   （相機朝下）
    clbrobot.set_servo_angle( 9, 50, 1)     # 頂部舵機 50   （相機朝上）

    # 小車前進三秒, 速度=3
    clbrobot.moveforward(50,2)              # 機器人前進
    clbrobot.t_stop(1)                      # 機器人停止
    # 小車後退三秒, 速度=3
    clbrobot.movebackward(50,2)             # 機器人後退
    clbrobot.t_stop(1)                      # 機器人停止
