import cv2
import numpy as np
import pandas as pd
date=input("Enter date- dd/mm/yyyy")
faceDetect = cv2.CascadeClassifier("C:/Users/gshik/Desktop/haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("C:/Users/gshik/Desktop/trainningData.yml")
id = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

df=pd.read_csv("C:/Users/gshik/Desktop/at1.csv")

while (True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    faces = faceDetect.detectMultiScale(gray, 2, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        id, conf = rec.predict(gray[y:y + h, x:x+w])
        id1 = df['Name'][df.Id==id]
        df[date][df.Id==id]='P'
        df.to_csv("C:/Users/gshik/Desktop/at1.csv",index=False)
        cv2.putText(img, str(id1), (x, y + h), font, 3, 255)
    cv2.imshow("Face", img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()