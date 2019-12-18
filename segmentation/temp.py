import cv2 as cv
import numpy as np
import copy
from matplotlib import pyplot as plt
img = cv.imread('todfy.png',0)
ys,xs= img.shape
img = cv.resize(img,(xs*5,ys*5), interpolation=cv.INTER_AREA)

template = cv.imread('yaaa.png',0)
w, h = template.shape[::-1]
template = cv.resize(template,(w*5,h*5), interpolation=cv.INTER_AREA)


# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED0']
meth = 'cv.TM_SQDIFF'
method = eval(meth)
# Apply template Matching
res = cv.matchTemplate(img,template,method)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
    print(min_loc)
    top_left = min_loc
    cv.circle(img,(top_left[0],top_left[1]),1, (255, 0, 0), -1)
else:
    print(max_loc)
    top_left = max_loc
    cv.circle(img,(top_left[0],top_left[1]),1, (255, 0, 0), -1)

bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(img,top_left, bottom_right, (0,0,255), 2)
plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle(meth)
plt.show()
cv.imwrite("temp.png",img)
