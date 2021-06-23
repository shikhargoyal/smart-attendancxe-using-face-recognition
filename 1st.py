import pandas as pd
df=pd.read_csv("C:/Users/gshik/Desktop/at1.csv")
import cv2
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('C:/Users/gshik/Desktop/haarcascade_frontalface_default.xml')

Id = input('Enter your id:')
Name=input('Enter your name:')
df2 = pd.DataFrame({"Id":[Id],"Name":[Name]})
df=pd.concat([df,df2]).drop_duplicates().reset_index(drop=True)
df.to_csv("C:/Users/gshik/Desktop/at1.csv",index=False)
sampleNum = 0
while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (144, 1, 1), 3)

        # incrementing sample number
        sampleNum = sampleNum + 1
        # saving the captured face in the dataset folder
        cv2.imwrite(
            "C:/Users/gshik/Desktop/face/user." + Id + '.' + str(sampleNum) + ".jpg",
            gray[y:y + h, x:x+w])

        cv2.imshow('frame', img)
    # wait for 100 miliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum > 20:
        break
cam.release()
cv2.destroyAllWindows()