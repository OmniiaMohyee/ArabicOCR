import cv2 as cv2
import numpy as np

img = cv2.imread('thresh.png')
# seen = img[:,200:250]
seen3= img[:,115:180]
# cv2.imwrite("seen3.png",seen3)
seen2= img[:,175:240]
cv2.imwrite("seen2.png",seen2)
