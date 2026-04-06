import cv2
import numpy as np

# Example for basic pixel based image manipulation:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html

# Task 1:   Implement some kind of noticeable image manipulation in this function
#           e.g. channel manipulation, filter you already know, drawings on the image etc.
def myFirstImageManipulation(img):
    result = img.copy()
    result[:,:,2] = 0 # set the blue channel to zero 
    return result



# Task 2:   Return the basic image properties to the console:
#           width, height,
#           the color of the first pixel of the image,
#           Color of the first pixel in the second row
#           Color of the first pixel in the second column
#           This function should work for images with three channels

def imageSize(img):
    height, width = img.shape[:2]
    return [width, height]

def getPixelColor(img):
    return [
        img[0,0].tolist(),  # first pixel
        img[1,0].tolist(),  # first pixel in second row
        img[0,1].tolist(),  # first pixel in second column
    ]

# Task 3:   Separate the given channels of a colour image in this function and return it as separate image
#           the separate image need three channels
#
def returnChannel(img, channel):
    b, g, r = cv2.split(img)
    
    channel_map = {0: 'b', 1: 'g', 2: 'r'}
    if isinstance(channel, int):
        channel = channel_map[channel]
    
    if channel == 'b':
        result = cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])
    elif channel == 'g':
        result = cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
    elif channel == 'r':
        result = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
    else:
        result = img.copy()  
    
    return result