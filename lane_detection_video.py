################################################################################################################
#                                   DETECTING LANE LINES IN A VIDEO
################################################################################################################



import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(frame):  #  FUNCTION FOR CANNY EDGE DETECTION
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(image, (5,5), 0) #Reduces noise in our image, makes it blur. (5x5) is the kernel used.
    canny_blur = cv2.Canny(blur, 50, 150)
    return canny_blur

def roi(frame): # FUNCTION FOR DEFINING REGION OF INTEREST
    image = canny(frame)
    height = image.shape[0]
    polygons = np.array([[(140, height), (910, height), (480, 290)]])  #The dimensions of the region of interest
    mask = np.zeros_like(image) # creates a mask of the triangle.
    cv2.fillPoly(mask, polygons, 255) #this overlaps the triangle to the mask created with totally white triangle.
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lanes(hough_lines, image): # FUNCTION FOR DISPLAYING LANE LINES
    mask_hough = np.zeros_like(image)
    if hough_lines is not None:
        for line in hough_lines:
            x1,y1,x2,y2 = line.reshape(4)  # Matrix containing the x & y coordinates of lane lines found using hough transform
            cv2.line(mask_hough, (x1,y1),(x2,y2), (255, 0, 0), 10)
    return mask_hough

def coordinates(image, line_parameters):  # FUNCTION FOR FINDING THE COORDINATES OF THE THE AVERAGED LANE LINES
    slope, intercept = line_parameters
    y1 = image.shape[0] #this is because we want our image to start in the bottom. which means the entire length of y axis should cover.
    y2 = int(y1*2.9 /5) + 6
    x1 = int((y1-intercept)/slope)  # as x = (y-b)/m
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(hough_lines, image): # FUNCTION FOR CALCULATING THE AVERAGE LANE LINES FROM A SERIES OF LINES
    left_fit = []  #contains the coordinates of the lines in the left lane lines
    right_fit = [] #contains the coordinates of the lines in the right lanelines
    for line in hough_lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)  #this fits first degree polynomial which will be a linear function of y = mx+c and return a vector of points that describe the slopee in y intercepts
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)  #CONTAINS MATRIX OF AVERAGE VALUES OF HOUGH TRANSFORM LINES OF LEFT LANE
    right_fit_average = np.average(right_fit, axis = 0)  #CONTAINS MATRIX OF AVERAGE VALUES OF HOUGH TRANSFORM LINES OF LEFT LANE
    left_line = coordinates(image, left_fit_average)
    right_line = coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

cap = cv2.VideoCapture("solidWhiteRight.mp4")  # TO READ THE VIDEO
while(cap.isOpened()):
    _, frame = cap.read()  # MAKES THE VIDEO INTO INDIVIDUAL VIDEO FRAMES AND FRAME ACTS AS IMAGES
    mask_image = roi(frame)
    hough_lines = cv2.HoughLinesP(mask_image, 2, np.pi/180, 50, np.array([]), minLineLength=20, maxLineGap=5)
    average_lines = average_slope_intercept(hough_lines, frame)
    lane_lines = display_lanes(average_lines, frame)
    final_image = cv2.addWeighted(frame, 0.8, lane_lines, 1, 1)
    cv2.imshow("canny", final_image)
    if cv2.waitKey(24) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
