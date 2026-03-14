import cv2
import numpy as np
import Utilities



def applyLUT(img, lut):#
    result = img.copy()
    
    return result.astype(np.uint8)


def equalizeHistogram(img):
    result = img.copy()
    return result, lut

def findMinMaxPos(histogram):
    minPos = 0
    maxPos = 255
    return minPos, maxPos

def stretchHistogram(img):
    result = img.copy()
    lut = Utilities.create_identity_lut()


    return result, lut

def calculateHistogram(img, nrBins):
    # create histogram vector
    histogram = np.zeros([nrBins], dtype=int)

    return histogram

def apply_log(img):
    lut = Utilities.create_identity_lut()


    return applyLUT(img, lut), lut

def apply_exp(img):
    lut = Utilities.create_identity_lut()


    return applyLUT(img, lut), lut

def apply_inverse(img):
    lut = Utilities.create_identity_lut()

    return applyLUT(img, lut), lut

def apply_threshold(img, threshold):
    lut = Utilities.create_identity_lut()

    return applyLUT(img, lut), lut


def apply_contrast_sigmoid(img, factor):
    lut =  np.linspace(0, 1, 256)

    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return applyLUT(img, lut), lut

def apply_contrast(img, factor):
    lut = Utilities.create_identity_lut()

    return applyLUT(img, lut), lut


def apply_exposure(img, ev):
    lut = Utilities.create_identity_lut()

    return applyLUT(img, lut), lut