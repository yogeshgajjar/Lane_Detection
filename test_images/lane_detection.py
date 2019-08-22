import cv2
import numpy as np
import matplotlib.pyplot as plt
#

# Converting it into greyscale

# def image_rgb():
#     #Normal RGB view
#     image_rgb = cv2.imread('solidWhiteCurve.jpg')
#     return image_rgb;
#     # cv2.imshow('img', image_rgb)
#     # cv2.waitKey(0)

def image_gray():
    image_grey = cv2.imread('whiteCarLaneSwitch.jpg', 0)
    return image_grey
    #cv2.imshow('img', image_grey)
    #cv2.waitKey(0)

def canny():
    image = image_gray()
    blur = cv2.GaussianBlur(image, (5,5), 0) #Reduces noise in our image, makes it blur. (5x5) is the kernel used.
    canny_blur = cv2.Canny(blur, 50, 150)
    return canny_blur

def roi():
    image = canny()
    height = image.shape[0]
    polygons = np.array([[(140, height), (910, height), (480, 290)]])  #870 #140
    mask = np.zeros_like(image) # creates a mask of the triangle.
    cv2.fillPoly(mask, polygons, 255) #this overlaps the triangle to the mask created with totally white triangle.
    masked_image = cv2.bitwise_and(image, mask)
    #cv2.imshow("int", masked_image)
    #cv2.waitKey(0)
    return masked_image

def display_lanes(hough_lines, image):
    #image = image_gray()
    mask_hough = np.zeros_like(image)
    if hough_lines is not None:
        for line in hough_lines:
            #print(line)
            x1,y1,x2,y2 = line.reshape(4)
            print("display lines x& y", x1,x2,y1,y2)
            cv2.line(mask_hough, (x1,y1),(x2,y2), (255, 0, 0), 10)
    return mask_hough

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    print(slope, "slope")
    y1 = image.shape[0] #this is because we want our image to start in the bottom. which means the entire length of y axis should cover.
    y2 = int(y1*2.9 /5) + 6
    print("New y", y1,y2)
    x1 = int((y1-intercept)/slope)  # as x = (y-b)/m
    x2 = int((y2-intercept)/slope)
    print("New x", x1,x2)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(hough_lines, image):
    left_fit = []  #contains the coordinates of the lines in the left lane lines
    right_fit = [] #contains the coordinates of the lines in the right lanelines
    for line in hough_lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)  #this fits first degree polynomial which will be a linear function of y = mx+c and return a vector of points that describe the slopee in y intercepts
        print("Parameters:", parameters)  #slope is parameters[0] and intercept is parameters[1]
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    print("left_fit_average", left_fit_average)
    print("right fit average", right_fit_average)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    print("Left line:", left_line)
    print("Right line:", right_line)
    return np.array([left_line, right_line])

original_image = cv2.imread('whiteCarLaneSwitch.jpg')
# plt.imshow(original_image)
# plt.show()
#cv2.imshow("ori", canny())
mask_image = roi()
hough_lines = cv2.HoughLinesP(mask_image, 2, np.pi/180, 50, np.array([]), minLineLength=20, maxLineGap=5)
average_lines = average_slope_intercept(hough_lines, original_image)
lane_lines = display_lanes(average_lines, original_image)
final_image = cv2.addWeighted(original_image, 0.8, lane_lines, 1, 1)
cv2.imwrite("finalimage.jpg", final_image)
cv2.imshow("canny", final_image)
cv2.waitKey(0)
