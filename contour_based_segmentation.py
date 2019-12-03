
import cv2

import numpy as np
 
im_name = "line1.png"


## SKEW DETECTION.
#1- Binarizing the image.
im_name = "test.png"
image = cv2.imread(im_name)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("thresh.png", thresh)

# Detecting the skew angle using column_stack.
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle
#Rotating the image.
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
#	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#Removing noise. should be replaced by another technique in case of arabic.
#blur = cv2.GaussianBlur(rotated,(5,5),0)


# show the output image
print("[INFO] angle: {:.3f}".format(angle))
cv2.imwrite("Rotated.png",rotated)
rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
rotated_gray = cv2.bitwise_not(rotated_gray)





