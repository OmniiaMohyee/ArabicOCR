import cv2 as cv
import numpy as np
import copy
from matplotlib import pyplot as plt
img = cv.imread('cnt/contoured.0.png')
(xs , ys , _)= img.shape
print(img.shape)
# img = cv.resize(img,(xs*5,ys*5), interpolation=cv.INTER_AREA)
template = cv.imread('cnt/seen2.png')
w, h ,_ = template.shape
print(template.shape)

# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED0']
meth = 'cv.TM_SQDIFF'
method = eval(meth)
# Apply template Matching
res = cv.matchTemplate(img,template,method)
threshold = 0.99
loc = np.where( res >= threshold)
print(len(loc))
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
print(min_val, max_val)
# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
# if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
print(min_loc)
top_left = min_loc
cv.circle(img,(top_left[0],top_left[1]),1, (255, 0, 0), -1)
# else:
#     print(max_loc)
#     top_left = max_loc
#     cv.circle(img,(top_left[0],top_left[1]),1, (255, 0, 0), -1)

bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(img,top_left, bottom_right, (0,0,255), 1)
cv.imwrite("img.png",img)

plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle(meth)
# plt.show()
