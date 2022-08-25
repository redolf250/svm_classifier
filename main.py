import cv2
from cv2 import VideoCapture
import numpy as np
import pickle
import dlib


cap = cv2.VideoCapture(0)

#detector = dlib.get_frontal_face_dector()

classifier = cv2.CascadeClassifier('D:\\Data\\work\\haarcascade_righteye_2splits.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_copy = frame.copy()
    detect = classifier.detectMultiScale(gray,1.2,1)
    for (x, y, w, h) in detect:
        cv2.rectangle(frame,(x, y), (x+w,y+h),(0,255,0),2)
    cv2.imshow('frame', np.hstack((frame_copy,frame)))
    if cv2.waitKey(20) & 0xff==ord('q'):
        break
cap.release()  
cv2.destroyAllWindows