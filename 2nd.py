#train data
import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create()
path='C:/Users/gshik/Desktop/face'

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(100) # for 10 milliseconds wait......
    return np.array(IDs), faces

Ids, faces=getImagesWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.save('C:/Users/gshik/Desktop/trainningData.yml')
cv2.destroyAllWindows()