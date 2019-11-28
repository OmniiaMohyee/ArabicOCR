
import cv2

import numpy as np
 


## SKEW DETECTION.
#1- Binarizing the image.
im_name = "capr1.png"
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


#counting number of zeros in each row.
hist = np.count_nonzero(rotated_gray, axis=1)
i = 0
while(i<len(hist)): 
    count = 0
    if(hist[i] == 0):
        j = 0
        while(i+j <len(hist)):
            if(hist[i+j]==0):
                j+=1
                count +=1
            else:
                break
        rotated_gray[i+int(count/2),:] = 255
        i+=j
    else:
        i+=1
cv2.imwrite("Rotated.png",rotated_gray)
line = 1
hist2 = np.count_nonzero(rotated_gray == 0 , axis=1)
#print(hist2)

bounds = []

for i in range(len(hist2)):
    if (hist2[i] == 0):
        bounds.append(i)
print(bounds)
for i in range(len(bounds)-1):
    result = rotated_gray[bounds[i]+1:bounds[i+1],:]
    cv2.imwrite("line"+str(line)+".png",result)
    line+=1

##############################################

im_name = "line1.png"
import cv2

import numpy as np
 


## SKEW DETECTION.
#1- Binarizing the image.
im_name = "capr1.png"
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


#counting number of zeros in each row.
hist = np.count_nonzero(rotated_gray, axis=1)
i = 0
while(i<len(hist)): 
    count = 0
    if(hist[i] == 0):
        j = 0
        while(i+j <len(hist)):
            if(hist[i+j]==0):
                j+=1
                count +=1
            else:
                break
        rotated_gray[i+int(count/2),:] = 255
        i+=j
    else:
        i+=1
cv2.imwrite("Rotated.png",rotated_gray)
line = 1
hist2 = np.count_nonzero(rotated_gray == 0 , axis=1)


bounds = []

for i in range(len(hist2)):
    if (hist2[i] == 0):
        bounds.append(i)
print(bounds)
for i in range(len(bounds)-1):
    result = rotated_gray[bounds[i]+1:bounds[i+1],:]
    cv2.imwrite("line"+str(line)+".png",result)
    line+=1

##############################################




for i in range(line):
    im_name = "line"+str(i+1)+".png"

    image = cv2.imread(im_name)

    hist3 = np.count_nonzero(image, axis=0)
    k = 0
    while(k<len(hist3)): 
        count = 0
        if(hist3[k].all() == 0):
            j = 0
            while(k+j <len(hist3)):
                if(hist3[k+j].all()==0):
                    j+=1
                    count +=1
                else:
                    break
            if(count>1):
                image[:,k+int(count/2)] = 255
            k+=j
        else:
            k+=1

    cv2.imwrite("line"+str(i+1)+".png",image)