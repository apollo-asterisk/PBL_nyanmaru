import pigpio 
import time

SERVO_PIN_1 = 13 
SERVO_PIN_2 = 12 
pi = pigpio.pi()

try: 
    while True: 
        pi.hardware_PWM(SERVO_PIN_1, 50, int(1000000*1.45/20)) 
#         time.sleep(1) 
        pi.hardware_PWM(SERVO_PIN_2, 50, int(1000000*1.45/20)) 
        time.sleep(1) 
        pi.hardware_PWM(SERVO_PIN_1, 50, int(1000000*2.4/20)) 
#         time.sleep(1) 
        pi.hardware_PWM(SERVO_PIN_2, 50, int(1000000*2.4/20)) 
        time.sleep(1) 
        pi.hardware_PWM(SERVO_PIN_1, 50, int(1000000*0.5/20)) 
#         time.sleep(1) 
        pi.hardware_PWM(SERVO_PIN_2, 50, int(1000000*0.5/20)) 
        time.sleep(1) 
except KeyboardInterrupt: 
    pass 

pi.hardware_PWM(SERVO_PIN_1, 50, 0) 
pi.hardware_PWM(SERVO_PIN_2, 50, 0) 
pi.stop()
