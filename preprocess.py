


import cv2
import numpy as np
import copy
 
def read_image(image_name):
    image = cv2.imread(image_name)
    return image

def binarize(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_gray = cv2.bitwise_not(image_gray)
    image_thresholded = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    return image_thresholded

#TODO: Add some function for noise removal.

def fix_skew(image,image_not_bin):
    coords = np.column_stack(np.where(image > 0))
    angle  = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = - (90 + angle)
    else:
        angle = - angle
    
    (h,w) = image.shape[:2]
    center = ( w//2 , h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode= cv2.BORDER_REPLICATE)
    rotated_not_bin = cv2.warpAffine(image_not_bin, M, (w,h), flags=cv2.INTER_CUBIC, borderMode= cv2.BORDER_REPLICATE)
    return rotated,rotated_not_bin



# folder = "./images/" 

# PIPELINE : read -> binarize -> fix_skew -> segment lines -> segment words.


def preproc(image_path):
    img = read_image(image_path)
    binarized_img = binarize(img)
    skew_fixed,skew_fixed_not_bin = fix_skew(binarized_img,img)
    #add here any other preprocessing steps
    return skew_fixed, skew_fixed_not_bin

