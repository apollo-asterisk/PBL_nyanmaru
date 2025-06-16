#coding:utf-8

import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2
import cv2
import numpy as np
import os

# --- モーター制御セットアップ ---
GPIO.setmode(GPIO.BOARD)
AIN1 = 8
AIN2 = 10
PWMA = 12
BIN1 = 22
BIN2 = 24
PWMB = 26

GPIO.setup(AIN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(AIN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PWMA, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BIN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BIN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PWMB, GPIO.OUT, initial=GPIO.LOW)

p_a = GPIO.PWM(PWMA, 100)
p_b = GPIO.PWM(PWMB, 100)
p_a.start(100) #
p_b.start(100) #速度を変えるとこ

def func_brake():
    GPIO.output(AIN1, GPIO.HIGH)
    GPIO.output(AIN2, GPIO.HIGH)
    GPIO.output(BIN1, GPIO.HIGH)
    GPIO.output(BIN2, GPIO.HIGH)

def func_right():
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(AIN2, GPIO.HIGH)
    GPIO.output(BIN1, GPIO.HIGH)
    GPIO.output(BIN2, GPIO.LOW)

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
    motor_state = None  # "right"または"brake"
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
            # 顔が見つかったら停止
            if motor_state != "brake":
                func_brake()
                motor_state = "brake"
        else:
            # 顔がなければ回転
            if motor_state != "right":
                func_right()
                motor_state = "right"

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
    picam2.stop()
    cv2.destroyAllWindows()
    print("End of program")
