import cv2
import sys
import time
faceCascade = cv2.CascadeClassifier('Hand.Cascade.1.xml')

video_capture = cv2.VideoCapture(0)

img_counter = 0
start = time.time()
count=0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    c=1
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'P{c}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # Display the resulting frame
        c+=1
    cv2.imshow('HandDetection', frame)

    if k%256 == 27: #ESC Pressed
        break
    count+=1
end = time.time()
seconds = end - start
print("Time taken : {0} seconds".format(seconds))
fps  = count / seconds
print ("Estimated frames per second : {0}".format(fps))
cv2.waitKey()
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()