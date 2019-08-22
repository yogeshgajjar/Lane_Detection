# lane_detection

Finding Lane lines on the Road

![Image of Lane Line Detection] (test_images/finalimage.jpg)

## Overview

When we drive, we use our eyes and brain to decide where to go. The lane lines acts as a constant reference for where to steer the vehicle. In Autonomous Vehicles, lane lines detection is essential part of letting the car know about it's desired lane trajectory.

## Concept

The lane line detection uses two essential concepts i.e Canny Edge Detection and Hough Transform.

1. Canny Edge Detection
  - Detects edges in a picture by finding the intensity of gradient of the image.
  - To detect edges, image is converted to grayscale image which turns easy for gradient detection.
  - OpenCV function : cv2.Canny()

2. Region of Interest
  - The lane lines in a 2-D image appears to be in a form of a triangle, which in turn becomes the ROI of our image.
  - Our main aim is to see the lane lines, so we use a masked image(black color) with white color triangle at the of the same dimension as our image.
  - We superimpose the black mask with white color ROI i.e. the triangle using Bitwise AND operation.

3. Hough Transform
  - Transform used to detect straight lines.
  - Uses polar coordinate to detect a point in the space. A straight line joining all the points gives a point in the Hough Transform space.
  - OpenCV function - cv2.HoughLinesP()

## Algorithm

The algorithm includes

- Filtering the image using GaussianBlur. Initial step for Canny Edge detection
- Use Canny Edge Detection method to convert the image to detect edges in the image.
- Define Region of Interest. Use a masked image of the same dimension as of the original image and superimpose it on the original image.
- Use Hough Transform to detect straight lines on the lane.
- Average the straight lines falling on the Region of Interest and make it smooth.

## Stack

- OpenCV for image processing
- Python
