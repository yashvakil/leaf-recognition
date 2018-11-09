import cv2
import numpy as np
from scipy import ndimage
import math
from skimage import feature
from ImagePreProcessing import *

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")

        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        # hist /= (hist.sum() + eps)
        return hist


def obtainHu(image):
    image = cv2.bitwise_not(image)
    # cv2.imshow("Bitwise Not", image)
    huInvars = cv2.HuMoments(cv2.moments(image)).flatten()  # Obtain hu moments from normalised moments in an array
    huInvars = -np.sign(huInvars) * np.log10(np.abs(huInvars))
    # huInvars /= huInvars.sum()
    return huInvars


def obtainHuMoments(image_path, image):  # Obtains hu moments from image at path location
    hu = obtainHu(image)
    image = cv2.imread(image_path, 0)
    lbp = LocalBinaryPatterns(24, 8)
    hist = lbp.describe(image)

    return hu, hist


def leafLW(image):
    l,w = image.shape[:2]
    return l-20,w-20

def findLeafContour(contours):
    maxArea = 0
    contour_index = 0;
    for i in range(1,len(contours)):
        area  =  cv2.contourArea(contours[i])
        if area > maxArea:
              maxArea = area
              contour_index = i
    return contour_index

def leafAP(image):
    value=230;
    retval, img = cv2.threshold(image, value, 255, type=cv2.THRESH_BINARY)
    _, contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    while(len(contours)<=1):
        value = value-5
        retval, img = cv2.threshold(image, value, 255, type=cv2.THRESH_BINARY)
        _, contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    leaf_contour_index = findLeafContour(contours)
    leaf_contour = contours[leaf_contour_index]
    cv2.drawContours(img, contours, leaf_contour_index, (0, 0, 255), 4)
    # cv2.imshow('contours', img)
    area = cv2.contourArea(leaf_contour)
    perimeter = cv2.arcLength(leaf_contour, True)
    return area, perimeter

def getLWAP(image_path):
    img = cv2.imread(image_path)
    while img.shape[1]>=800 or img.shape[2]>=800:
        img = cv2.resize(img, None, fx=0.9, fy=0.9)

    angle = rotationAngle(img)
    img = rotateImage(img, angle)
    img = croppedImage(img)
    length, width = leafLW(img)
    # print(length, width)
    # img = resizeImage(img, 800)
    area, perimeter = leafAP(img)
    # print(area, perimeter)
    return img,length,width,area,perimeter

def getAFR(length,width,area,perimeter):
    aspect_ratio = length/width
    form_factor = 4*3.14159265358*area / perimeter
    rectangularity = length*width / area
    return  aspect_ratio,form_factor,rectangularity

def getAllFeatures(image_path):
    img, length, width, area, perimeter = getLWAP(image_path)
    aspect_ratio, form_factor, rectangularity = getAFR(length, width, area, perimeter)
    hu, hist = obtainHuMoments(image_path,img)
    return length,width,area,perimeter,aspect_ratio,form_factor,rectangularity,hu,hist

cv2.waitKey(0)
cv2.destroyAllWindows()