# coding:utf-8
import numpy as np;
import cv2;

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# カメラ
cap = cv2.VideoCapture(0)
cap.set(3, 640) # 横サイズ
cap.set(4, 480) # 縦サイズ

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if ret == False:
        break

    # 顔を検知
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 5)

    cv2.imshow('Face Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
