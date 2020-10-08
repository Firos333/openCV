import cv2 
import imutils 
import time
# Initializing the HOG person 
# detector 
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 

start = time.time()
count=0

cap = cv2.VideoCapture('VID_20200925_180449.mp4') 
while cap.isOpened(): 
    # Reading the video stream 
    ret, image = cap.read() 
    c = 1
    if ret: 
        image = imutils.resize(image,  
                               width=min(1400, image.shape[1])) 
        (regions, _) = hog.detectMultiScale(image, 
                                            winStride=(4, 4), 
                                            padding=(4, 4), 
                                            scale=1.05) 
   
        # Drawing the regions in the  
        # Image 
        for (x, y, w, h) in regions: 
            cv2.rectangle(image, (x, y), 
                          (x + w, y + h),  
                          (0, 0, 255), 2) 
            cv2.putText(image, f'P{c}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            c += 1
        # Showing the output Image 
        cv2.putText(image, f'Total Persons : {c - 1}', (20, 450), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255,0), 2)
        cv2.imshow("Image", image) 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    else: 
        break
    count+=1

end = time.time()
seconds = end - start
print("Time taken : {0} seconds".format(seconds))
fps  = count / seconds;
print ("Estimated frames per second : {0}".format(fps))

cap.release() 
cv2.destroyAllWindows() 