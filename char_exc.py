import cv2
import numpy as np

#1- Binarizing the image.
im_name = "word.png"
image = cv2.imread(im_name)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#cv2.imshow("thresh.png", thresh)
#cv2.imwrite("word_thresh.png",thresh)
vert_hist = np.count_nonzero(thresh,axis = 0)
print(vert_hist)
i = 0
while(i<len(vert_hist)):
    count = 0
    if(vert_hist[i] == 0):
        j = 0
        while(i+j < len(vert_hist)):
            if(vert_hist[i+j] == 0):
                j+=1
                count += 1
            else:
                break
        if(count>2):
            thresh[:,i+int(count/2)] = 255
        i+=j
    else:
        i+=1
cv2.imwrite("word_seg.png",thresh)

######################################
# im_name = "char.png"
# image = cv2.imread(im_name)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.bitwise_not(gray)
# thresh = cv2.threshold(gray, 0, 255,
# 	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#im_hist = np.count_nonzero(thresh,axis=0)
#print("IMAGE HIISSSSTTT",im_hist)
stroke = sum(vert_hist)/len(vert_hist)
print(stroke)
#for i in range(len(vert_hist)):
#   vert_hist[i] -= stroke
stroke = 15
for i in range(len(vert_hist)):
   if(vert_hist[i] - stroke < 3 and vert_hist[i] - stroke > -3 ):
       #print(i)
       image[:,i] = 255
cv2.imwrite("CHAAAAR.png",image)
# print(len(line_hist))
# #index = np.where(line_hist == np.amax(line_hist))
# #print(index)
# #line[:,index] = 255
# maxi = line_hist[0][0]
# ind = 0
# print(maxi)
# print(ind)
# for i in range(len(line_hist)):
#     if line_hist[i][0] > maxi:
#         ind = i
#         maxi = line_hist[i][0]
#         print(ind,maxi)

# print(ind)
# line[ind,:] = 255
# cv2.imwrite("bl_line.png",line)
