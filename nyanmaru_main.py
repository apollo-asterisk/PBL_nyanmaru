import time
import pigpio
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import cv2
import numpy as np

# モーターセットアップ
GPIO.setmode(GPIO.BOARD)
AIN1 = 38
AIN2 = 40
PWMA = 12
BIN1 = 22
BIN2 = 24
PWMB = 26

SERVO_PIN_1 = 13
SERVO_PIN_2 = 12
BUTTON_PIN = 31

pi = pigpio.pi()

GPIO.setup(AIN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(AIN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PWMA, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BIN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BIN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PWMB, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

p_a = GPIO.PWM(PWMA, 50)
p_b = GPIO.PWM(PWMB, 50)
p_a.start(100)
p_b.start(100)

running = False

def button_callback(channel):
    global running
    running = not running
    if running:
        print("start")
    else:
        print("stop")
        func_stop()

GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=button_callback, bouncetime=300)

def func_forward():
    GPIO.output(AIN1, GPIO.HIGH)
    GPIO.output(AIN2, GPIO.LOW)
    GPIO.output(BIN1, GPIO.LOW)
    GPIO.output(BIN2, GPIO.HIGH)

def func_right():
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(AIN2, GPIO.HIGH)
    GPIO.output(BIN1, GPIO.LOW)
    GPIO.output(BIN2, GPIO.HIGH)

def func_stop():
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(AIN2, GPIO.LOW)
    GPIO.output(BIN1, GPIO.LOW)
    GPIO.output(BIN2, GPIO.LOW)

def move_ears():
    for _ in range(3):
        if not running:
            break
        pi.hardware_PWM(SERVO_PIN_1, 50, int(1000000*1.4/20))
        pi.hardware_PWM(SERVO_PIN_2, 50, int(1000000*0.5/20))
        time.sleep(0.5)
        pi.hardware_PWM(SERVO_PIN_1, 50, int(1000000*0.5/20))
        pi.hardware_PWM(SERVO_PIN_2, 50, int(1000000*1.4/20))
        time.sleep(0.5)
    pi.hardware_PWM(SERVO_PIN_1, 50, 0)
    pi.hardware_PWM(SERVO_PIN_2, 50, 0)

# 超音波センサ
TRIG_PIN = 8
ECHO_PIN = 10
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
speed_of_sound = 34370

def get_distance():
    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, GPIO.LOW)
    timeout = time.time() + 0.04
    while not GPIO.input(ECHO_PIN):
        if time.time() > timeout:
            return -1
    t1 = time.time()
    timeout = time.time() + 0.04
    while GPIO.input(ECHO_PIN):
        if time.time() > timeout:
            return -1
    t2 = time.time()
    return (t2 - t1) * speed_of_sound / 2

# 顔認識
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']
width, height = 640, 480
minW = 0.1 * width
minH = 0.1 * height

# カメラ初期化
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
picam2.configure(preview_config)
picam2.start()

try:
    motor_state = None

    while True:
        if running:
            frame = picam2.capture_array()
            img = cv2.flip(frame, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))

            if len(faces) > 0:
                distance = get_distance()
                print(f"Distance: {distance:.1f}cm")
                if 0 < distance < 25:
                    if motor_state != "stop":
                        func_stop()
                        print("25cm未満：停止して耳を動かす")
                        motor_state = "stop"
                        move_ears()
                else:
                    if motor_state != "forward":
                        func_forward()
                        print("前進")
                        motor_state = "forward"
            else:
                if motor_state != "right":
                    func_right()
                    print("顔なし：回転")
                    motor_state = "right"

            # 画面表示
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                label = names[id] if confidence < 100 else "unknown"
                conf_str = f"  {round(100 - confidence)}%"
                cv2.putText(img, str(label), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, conf_str, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            cv2.imshow('camera', img)

        else:
            if motor_state != "stop":
                func_stop()
                motor_state = "stop"

        if cv2.waitKey(10) & 0xFF == 27:  # ESCで終了
            break

finally:
    p_a.stop()
    p_b.stop()
    GPIO.cleanup()
    pi.hardware_PWM(SERVO_PIN_1, 50, 0)
    pi.hardware_PWM(SERVO_PIN_2, 50, 0)
    pi.stop()
    picam2.stop()
    cv2.destroyAllWindows()
    print("End of program")
