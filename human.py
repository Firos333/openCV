import cv2
import time
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascades/hogcascade_pedestrians.xml')

# Read the input image
img = cv2.imread('street_trees_nottingham-36-edit-2000x1335.jpeg')
start = time.time()
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
cv2.imshow('img', img)
end = time.time()
seconds = end - start
print("Time taken : {0} seconds".format(seconds))
fps  = 1 / seconds
print ("Estimated frames per second : {0}".format(fps))
cv2.waitKey()

cv2.imwrite('color.jpg', img )
print("Image is saved color")
cv2.destroyAllWindows()