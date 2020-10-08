import cv2 
import imutils 
import time
# Initializing the HOG person 
# detector 
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
cap = cv2.VideoCapture(0) 
start = time.time()
counter=0
num_frames = 4
# Reading the Image 
while(1):
    ret, image = cap.read()
    
    # Resizing the Image 
    # image = imutils.resize(image, 
                        # width=min(800, image.shape[1])) 
    
    # Detecting all the regions in the  
    # Image that has a pedestrians inside it 
    (regions, _) = hog.detectMultiScale(image,  
                                        winStride=(4, 4), 
                                        padding=(4, 4), 
                                        scale=1.05) 
    
    # Drawing the regions in the Image 
    for (x, y, w, h) in regions: 
        cv2.rectangle(image, (x, y),  
                    (x + w, y + h),  
                    (0, 0, 255), 2) 
    
    # Showing the output Image 
    cv2.imshow("Image", image) 
    k = cv2.waitKey(1) & 0xff
    if k == 27: 
        break 
    counter=counter+1

end = time.time()
seconds = end - start
print("Time taken : {0} seconds".format(seconds))
fps  = counter / seconds
print ("Estimated frames per second : {0}".format(fps))
cv2.waitKey(0) 
   
cv2.destroyAllWindows() 