import time
import pigpio
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import cv2
import numpy as np

# --- GPIO & PWM setup ---
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
TRIG_PIN = 8
ECHO_PIN = 10

# --- 初期化 ---
pi = pigpio.pi()
GPIO.setup(AIN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(AIN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PWMA, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BIN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BIN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PWMB, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

p_a = GPIO.PWM(PWMA, 50)
p_b = GPIO.PWM(PWMB, 50)
p_a.start(100)
p_b.start(100)

running = False
speed_of_sound = 34370  # cm/s

# --- ボタンコールバック ---
def button_callback(channel):
    global running
    running = not running
    print("== ボタンが押されました ==")
    print("start" if running else "stop")
    if not running:
        func_stop()

GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=button_callback, bouncetime=800)

# --- モーター制御関数 ---
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

# --- サーボ制御関数 ---
def move_ears():
    print("== 耳を動かします ==")
    for _ in range(3):
        if not running:
            break
        pi.hardware_PWM(SERVO_PIN_1, 50, int(1000000 * 1.4 / 20))
        pi.hardware_PWM(SERVO_PIN_2, 50, int(1000000 * 0.5 / 20))
        time.sleep(0.5)
        pi.hardware_PWM(SERVO_PIN_1, 50, int(1000000 * 0.5 / 20))
        pi.hardware_PWM(SERVO_PIN_2, 50, int(1000000 * 1.4 / 20))
        time.sleep(0.5)
    pi.hardware_PWM(SERVO_PIN_1, 50, 0)
    pi.hardware_PWM(SERVO_PIN_2, 50, 0)

# --- 距離測定 ---
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
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']

width, height = 800, 600
minW = int(0.1 * width)
minH = int(0.1 * height)

# --- カメラ初期化 ---
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
picam2.configure(preview_config)
picam2.start()

# --- メインループ ---
try:
    motor_state = None
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)

    while True:
        frame = picam2.capture_array()
        img = cv2.flip(frame, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(minW, minH))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            label = names[id] if confidence < 100 else "unknown"
            conf_str = f"{round(100 - confidence)}%"
            cv2.putText(img, label, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, conf_str, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        if running:
            if len(faces) > 0:
                distance = get_distance()
                if distance != -1 and 0 < distance < 15:
                    if motor_state != "stop":
                        func_stop()
                        print(f"距離: {distance:.1f}cm → 停止して耳を動かします")
                        motor_state = "stop"
                        move_ears()
                else:
                    if motor_state != "forward":
                        func_forward()
                        if distance == -1:
                            print("距離測定失敗 → それでも前進")
                        else:
                            print(f"距離: {distance:.1f}cm → 前進")
                        motor_state = "forward"
            else:
                if motor_state != "right":
                    func_right()
                    print("顔が見つからないので回転")
                    motor_state = "right"
        else:
            if motor_state != "stop":
                func_stop()
                print("== 動作停止 ==")
                motor_state = "stop"

        if cv2.waitKey(100) & 0xFF == 27:
            break

# --- 終了処理 ---
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
