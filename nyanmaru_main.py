import os
os.system("sudo pigpiod")
import time
import pigpio
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import cv2
import numpy as np

# --- モーター制御セットアップ ---
GPIO.setmode(GPIO.BOARD)
AIN1 = 38
AIN2 = 40
PWMA = 12
BIN1 = 22
BIN2 = 24
PWMB = 26

# --- servo制御セットアップ ---
SERVO_PIN_1 = 13
SERVO_PIN_2 = 12
pi = pigpio.pi()

GPIO.setup(AIN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(AIN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PWMA, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BIN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BIN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PWMB, GPIO.OUT, initial=GPIO.LOW)

p_a = GPIO.PWM(PWMA, 100)
p_b = GPIO.PWM(PWMB, 100)
p_a.start(100)
p_b.start(100)

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
    # サーボを動かす（例：2.4ms→0.5ms→中央に戻す）
    pi.hardware_PWM(SERVO_PIN_1, 50, int(1000000*2.4/20))
    pi.hardware_PWM(SERVO_PIN_2, 50, int(1000000*2.4/20))
    time.sleep(0.5)
    pi.hardware_PWM(SERVO_PIN_1, 50, int(1000000*0.5/20))
    pi.hardware_PWM(SERVO_PIN_2, 50, int(1000000*0.5/20))
    time.sleep(0.5)
    # サーボを停止
    pi.hardware_PWM(SERVO_PIN_1, 50, 0)
    pi.hardware_PWM(SERVO_PIN_2, 50, 0)

# --- 超音波センサーセットアップ ---
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

# --- 顔認識セットアップ ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']
width, height = 640, 480
minW = 0.1 * width
minH = 0.1 * height

# Picamera2初期化
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"size": (width, height), "format": "RGB888"}
)
picam2.configure(preview_config)
picam2.start()

try:
    motor_state = None  # "right"または"forward"または"stop"
    stopped_for_face = False
    while True:
        frame = picam2.capture_array()
        img = cv2.flip(frame, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        if len(faces) > 0:
            # 顔が見つかった
            distance = get_distance()
            print(f"Distance: {distance:.1f}cm")
            if 0 < distance < 10:
                if motor_state != "stop":
                    func_stop()
                    print("10cm未満なので停止し耳を動かします")
                    move_ears()
                    motor_state = "stop"
                    stopped_for_face = True
            else:
                if motor_state != "forward":
                    func_forward()
                    print("前進")
                    motor_state = "forward"
                stopped_for_face = False
        else:
            # 顔がなければ回転
            if motor_state != "right":
                func_right()
                print("顔が見つからないので回転")
                motor_state = "right"
            stopped_for_face = False

        # 画面表示（オプション）
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 100:
                label = names[id]
                conf_str = "  {0}%".format(round(100 - confidence))
            else:
                label = "unknown"
                conf_str = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(label), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(conf_str), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        cv2.imshow('camera', img)

        if cv2.waitKey(10) & 0xff == 27:  # ESCで終了
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
