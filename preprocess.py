


import cv2
import numpy as np
import copy
 
def read_image(image_name):
    image = cv2.imread(image_name)
    return image

def binarize(image):
    # ys , xs, _ = image.shape
    # resized_img = cv2.resize(image,(xs*5,ys*5), interpolation=cv2.INTER_AREA)
    # cv2.imwrite("resized.png", resized_img)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_gray = cv2.bitwise_not(image_gray)
    image_thresholded = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    return image_thresholded

#TODO: Add some function for noise removal.

def fix_skew(image):
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
    return rotated



# folder = "./images/" 

# PIPELINE : read -> binarize -> fix_skew -> segment lines -> segment words.


def preproc(image_path):
    img = read_image(image_path)
    binarized_img = binarize(img)
    skew_fixed = fix_skew(binarized_img)
    #add here any other preprocessing steps
    clean_img = skew_fixed
    return clean_img




