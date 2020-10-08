import cv2 
import sys
import numpy
import time

num_frames = 120
face_cascade = cv2.CascadeClassifier('haarcascades/cascadG.xml') 
# hand_cascade = cv2.CascadeClassifier('Hand.Cascade.1.xml') 

cap = cv2.VideoCapture(0) 
start = time.time()
count=0
num_frames =100
while(1):
    ret, img = cap.read() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE) 
    for (x,y,w,h) in faces: 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 

    # hand = hand_cascade.detectMultiScale(gray,1.3, 5) 
    # for (ex,ey,ew,eh) in hand: 
    #     cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 

    cv2.imshow('img',img) 
    # fps  = 1 / (time.time() - start)
    # print(fps)
    k = cv2.waitKey(1) & 0xff
    if k == 27: 
        break 
    count=count+1

end = time.time()
seconds = end - start
print("Time taken : {0} seconds".format(seconds))
fps  =count / seconds
print ("Estimated frames per second : {0}".format(fps))

cap.release() 
cv2.destroyAllWindows() 
