import datetime
import math
import cv2
import numpy as np

# Global variables
width = 0
height = 0
EntryCounter = 0
ExitCounter = 0
MinimumContourArea = 3000  # This value is empirical. Adjust it according to your needs.
BinarizationThreshold = 70  # This value is empirical. Adjust it according to your needs.
ReferenceLinesOffset = 150  # This value is empirical. Adjust it according to your needs.

# Check if the detected body is entering the monitored area
def TestEntryIntersection(y, EntryLineYCoord, ExitLineYCoord):
    AbsoluteDifference = abs(y - EntryLineYCoord)	

    if AbsoluteDifference <= 2 and y < ExitLineYCoord:
        return 1
    else:
        return 0

# Check if the detected body is exiting the monitored area
def TestExitIntersection(y, EntryLineYCoord, ExitLineYCoord):
    AbsoluteDifference = abs(y - ExitLineYCoord)	

    if AbsoluteDifference <= 2 and y > EntryLineYCoord:
        return 1
    else:
        return 0

camera = cv2.VideoCapture(0)

# Force the camera to have a resolution of 640x480
camera.set(3, 640)
camera.set(4, 480)

FirstFrame = None

# Perform some initial frame readings before processing
# This is done to allow the camera to adjust to the lighting conditions
for i in range(0, 20):
    (grabbed, Frame) = camera.read()

while True:
    # Read the first frame and determine the image resolution
    (grabbed, Frame) = camera.read()
    height = np.size(Frame, 0)
    width = np.size(Frame, 1)

    # If it's not possible to obtain a frame, stop processing
    if not grabbed:
        break

    # Convert the frame to grayscale and apply a blur effect to enhance contours
    FrameGray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    FrameGray = cv2.GaussianBlur(FrameGray, (21, 21), 0)

    # If it's the first frame (or first iteration), initialize the background frame
    if FirstFrame is None:
        FirstFrame = FrameGray
        continue

    # Calculate the absolute difference between the first frame and the current frame (background subtraction)
    # Additionally, binarize the frame with the background subtracted
    FrameDelta = cv2.absdiff(FirstFrame, FrameGray)
    FrameThresh = cv2.threshold(FrameDelta, BinarizationThreshold, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate the binary frame to eliminate "holes" or white areas within detected contours
    # This way, detected objects will be considered as black masses
    # Find the contours after dilation
    FrameThresh = cv2.dilate(FrameThresh, None, iterations=2)
    _, contours, _ = cv2.findContours(FrameThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ContourCount = 0

    # Draw reference lines on the frame
    EntryLineYCoord = (height / 2) - ReferenceLinesOffset
    ExitLineYCoord = (height / 2) + ReferenceLinesOffset
    cv2.line(Frame, (0, EntryLineYCoord), (width, EntryLineYCoord), (255, 0, 0), 2)
    cv2.line(Frame, (0, ExitLineYCoord), (width, ExitLineYCoord), (0, 0, 255), 2)

    # Iterate through all the detected contours
    for contour in contours:
        # Ignore contours with a very small area
        if cv2.contourArea(contour) < MinimumContourArea:
            continue

        # For debugging purposes, count the number of detected contours
        ContourCount += 1    

        # Get the coordinates of the contour (actually, a rectangle that encompasses the entire contour) and
        # draw a rectangle around the contour
        (x, y, w, h) = cv2.boundingRect(contour)  # x and y are the coordinates of the upper-left vertex
                                                  # w and h are the width and height of the rectangle

        cv2.rectangle(Frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Determine the central point of the contour and draw a circle to indicate it
        CenterXCoord = (x + x + w) / 2
        CenterYCoord = (y + y + h) / 2
        ContourCenter = (CenterXCoord, CenterYCoord)
        cv2.circle(Frame, ContourCenter, 1, (0, 0, 0), 5)
        
        # Test the intersection of the contour centers with the reference lines
        # This way, we count which contours crossed which lines (in a specific direction)
        if TestEntryIntersection(CenterYCoord, EntryLineYCoord, ExitLineYCoord):
            EntryCounter += 1

        if TestExitIntersection(CenterYCoord, EntryLineYCoord, ExitLineYCoord):  
            ExitCounter += 1

    print("Detected contours: " + str(ContourCount))

    # Write the number of people who entered or exited the monitored area on the frame
    cv2.putText(Frame, "Entries: {}".format(str(EntryCounter)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 1), 2)
    cv2.putText(Frame, "Exits: {}".format(str(ExitCounter)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Original", Frame)
    cv2.waitKey(1)

# Release the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
