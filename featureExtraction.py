import numpy as np
import math
import cv2 

#read image
img = cv2.imread('t2214.png')
img_grey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
_, bw_img = cv2.threshold(img_grey,127,255,cv2.THRESH_BINARY) #convertt to binary
img_width, img_height = bw_img.shape

# 1.extract width and height
first_row = img_height
first_col = img_width
last_row = 0
last_col = 0

for i in range (img_height):
    for j in range(img_width):
        if (bw_img[i][j]==0):
            first_row = min(first_row,i)
            first_col = min(first_col,j)
            last_row = max (last_row,i)
            last_col = max (last_col,j)

cropped_img = bw_img[first_row:last_row,first_col:last_col]
new_width ,new_height = cropped_img.shape
# 2. extract black and white pixels
black_pixels = 0
white_pixels = 0
for i in range (new_height):
    for j in range(new_width):
        if (bw_img[i][j]==0):
            black_pixels +=1
        else:
            white_pixels +=1

cv2.imshow("r",cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()