//カメラをうごかすプログラム

'''
Simple Cam Test - BGR and Gray
    Create by pythonprogramming.net ==> See the tutorial here:
    https://pythonprogramming.net/loading-video-python-opencv-tutorial
Adapted by Marcelo Rovai - MJRoBot.org @8Feb18
'''

from picamera2 import Picamera2
import cv2

picam2=Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format":'BGR888',"size":(640,480)}))
picam2.start()

while True:
    frame=picam2.capture_array()
    cv2.imshow("Pi Camera",frame)
    if cv2.waitKey(1)==ord('q'):
        break
    
cv2.destroyAllWindows()
picam2.stop()
