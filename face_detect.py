import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
img = cv2.imread("C:/Users/SATYAM/PycharmProjects/Face_Detection/image/input.jpg")
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #to convert the image in gray
faces = face_cascade.detectMultiScale(gray_img,1.3,5) #to read the faces using the haar classifier
print(type(faces))
print(faces)

for x,y,w,h in faces:
    img = cv2.rectangle(gray_img,(x,y),(x+w,y+h),(255,0,0),3) #to draw the blue color (B,G,R) rectangle around the face of thickness 3
    resized = cv2.resize(img, (int(img.shape[1]/5),int(img.shape[0]/5)))
    cv2.imshow('Face Detection',resized)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()