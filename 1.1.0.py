# organize imports
import cv2
import imutils
import numpy as np

# global variables
bg = None
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg

    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)




if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5
    hand_cascade = cv2.CascadeClassifier('Hand.Cascade.1.xml') 
    # get the reference to the webcam
    frame = cv2.imread('Shahruk_Khan_20130529_1.jpg')

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    # (grabbed, frame) = img.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = hand_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x,y,w,h) in faces: 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w] 
        roi = frame[y:y+h, x:x+w]
        print(roi)
        if roi.any():
        # resize the frame
            frame = imutils.resize(frame, width=700)

            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # clone the frame
            clone = frame.copy()

            # get the height and width of the frame
            (height, width) = frame.shape[:2]
            # get the ROI
            # roi = frame[(x,y),(x+w,y+h)]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            # to get the background, keep looking till a threshold is reached
            # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (y+h, x+w)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

            # draw the segmented hand
                cv2.rectangle(clone, (x+w, y), (x, y+h), (0,255,0), 2)

                # increment the number of frames
                num_frames += 1

                # display the frame with segmented hand
                cv2.imshow("Video Feed", clone)

# camera.release()
cv2.destroyAllWindows()