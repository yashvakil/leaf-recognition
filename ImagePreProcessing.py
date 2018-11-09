import cv2
import numpy as np
from scipy import ndimage
import math

x1=y1=x2=y2=flag=angle=0

def mouseClickEvent(event,x,y,flags,param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        global flag,angle
        global x1,x2,y1,y2
        flag= flag+1
        if flag == 1:
            x1=x
            y1=y
            # print(x,y)
        elif flag == 2:
            x2=x
            y2=y
            # print(x,y)
            slope = ((y2-y1)/(x2-x1))
            angle = math.atan(slope)*180*7/22

def croppedImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    # cv2.imshow("Grey", gray)
    blur = cv2.medianBlur(gray, 5) # make the image blur
    # cv2.imshow("Blur", blur)
    retval, thresh_gray = cv2.threshold(blur, 200, 255, type=cv2.THRESH_BINARY) # threshold to get just the leaf
    # cv2.imshow("Thresh", thresh_gray)
    # find where the leaf is and make a cropped region
    points = np.argwhere(thresh_gray == 0)  # find where the black pixels are
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    min_x = min(x for (x, y) in points)
    min_y = min(y for (x, y) in points)
    max_x = max(x for (x, y) in points)
    max_y = max(y for (x, y) in points)
    crop = blur[min_y-10:max_y+10, min_x-10:max_x+10]  # create a cropped region of the blur image
    retval, thresh_crop = cv2.threshold(crop, 200, 255, type=cv2.THRESH_BINARY)
    # cv2.imshow('Thresh and Cropped', thresh_crop)
    return thresh_crop

def resizeImage(image, size):
    # cv2.imshow('Resized', cv2.resize(image, (size,size), interpolation=cv2.INTER_CUBIC))
    return cv2.resize(image, (size,size), interpolation=cv2.INTER_CUBIC)

def rotateImage(image, angle):
    # cv2.imshow('Rotated', ndimage.rotate(image, angle, cval=256))
    return ndimage.rotate(image, angle, cval=256)

def rotationAngle(image):
    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', mouseClickEvent)
    while (1):
        if cv2.waitKey(0): cv2.destroyAllWindows();break
    return angle

cv2.waitKey(0)
cv2.destroyAllWindows()