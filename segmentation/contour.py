
import cv2
import numpy as np
from scipy.signal import argrelextrema
import copy
from PIL import Image
## SKEW DETECTION.
#1- Binarizing the image.
im_name = "c1.png"
image = cv2.imread(im_name)
(ys , xs , _)= image.shape



image = cv2.resize(image,(xs*6,ys*6), interpolation=cv2.INTER_AREA)
cv2.imwrite("resized.png", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
out = open('out.txt','w')

bi = copy.copy(gray)
for i in range(xs):
	for j in range(ys):
		if(bi[i][j] > 60):     
			bi[i][j] = 255
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# thresh = cv2.blur(thresh,(2,2))
cv2.imwrite("thresh.png", thresh)
cv2.imwrite("bin.png", bi)
# thresh =bi
# base line

h_line_hist = np.count_nonzero(thresh,axis=1)
v_line_hist = np.count_nonzero(thresh,axis=0)

line_index = np.argmax(h_line_hist)
max = np.amax(h_line_hist)
# thresh[line_index,:] = 255
cv2.imwrite("bl_line.png",thresh)

# edges 

c1_edges = cv2.Canny(thresh,0,500)
cv2.imwrite("edged.png",c1_edges)

# countour 

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(x) for x in contours]
i = np.argmax(areas)
cnt = contours[i]
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

# print(cnt)

leftmost =tuple( cnt[cnt[:,:,0].argmin()][0])
rightmost =tuple (cnt[cnt[:,:,0].argmax()][0])
topmost = tuple( cnt[cnt[:,:,1].argmin()][0])
bottommost =tuple(cnt[cnt[:,:,1].argmax()][0])


image[rightmost[1],rightmost[0]] = (255,0,0)
image[leftmost[1],leftmost[0]] = (255,0,0)
image[topmost[1],topmost[0]] = (255,0,0)
image[bottommost[1],bottommost[0]] = (255,0,0)

threshold = 10
base_line = line_index 

# print(list(cnt)[0])
list = []
for i in range(0,len(cnt)):
    x = cnt[i][0][0]
    y = cnt[i][0][1]
    list.append((x,y))
# print(list)
right = list.index(rightmost)
left = list.index(leftmost)
top = list.index(topmost)
bottom = list.index(bottommost)
l=[]
print(base_line)
for i in range(0,len(cnt)):
	x = cnt[i][0][0]
	y = cnt[i][0][1]
	if y <= base_line :
		l.append([x,y])
cv2.drawContours(image, [np.asarray(l)], 0, (0,255,0), 1)
# cv2.drawContours(image, [cnt[0:interval2]], 0, (0,255,0), 1)

cv2.imwrite("contoured.png",image)

l.sort(key=lambda x:x[0])
list=l
# print(defects)
list_x=[]
list_y=[]
# list_x = list[:,0]
# list_y = list[:,1]
# indexes = np.unique(list_x, return_index=True)[1]
# list_x = [list_x[index] for index in sorted(indexes)]
last_y = 0
for i in range(len(list)):
	y= list[i][1]
	x= list[i][0]
	# if(y != last_y ):
	list_x.append(x)
	list_y.append(y)
	# last_y = y
	# cv2.circle(image,(int(x),int(y)), 1, (0, 0, 255), -1)


# minimas = argrelextrema(np.asarray(list_x), np.less)
# print(list_y)
maximas = (np.diff(np.sign(np.diff(np.asarray(list_y)))) > 0).nonzero()[0] + 1 # local min
minimas = (np.diff(np.sign(np.diff(np.asarray(list_y)))) < 0).nonzero()[0] + 1 # local min
# print([[list_x[i],list_y[i]] for i in maximas] )
# print([[list_x[i],list_y[i]] for i in minimas] )
max_min = [[list_x[i],list_y[i]] for i in maximas ]+[[list_x[i],list_y[i]] for i in minimas]

min = []
max = []

for m in minimas:
	x,y=[list_x[m],list_y[m]]
	if(y < base_line):
		continue
	# cv2.circle(image,(int(x),int(y)), 1, (255, 0, 0), -1)
	min.append(m)

for m in maximas:
	x,y=[list_x[m],list_y[m]]
	cv2.circle(image,(int(x),int(y)),1, (255, 255, 0), -1)
cv2.circle(image,(47,66),1, (0, 255, 255), -1)
cv2.circle(image,(60,60),1, (0, 255, 255), -1)
cv2.circle(image,(66,54),1, (0, 255, 255), -1)
cv2.circle(image,(77,65),1, (0, 255, 255), -1)
cv2.circle(image,(41,60),1, (0, 255, 255), -1)
cv2.circle(image,(36,60),1, (0, 255, 255), -1)
cv2.circle(image,(35,48),1, (0, 255, 255), -1)

# max_min.sort(key=lambda x:x[0])
print(max_min)
# print(max_min)
	# print(x,y)
	# cv2.circle(image,(int(x),int(y)), 1, (255, 0, 0), -1)
	# cv2.line(image,(int(x),int(y)),(int(x),int(y)+200),(0,255,255),2)
 

	# print(x,y)
	# cv2.circle(image,(int(x),int(y)), 1, (0, 0, 255), -1)
cv2.imwrite("contoured.png",image)

selected_mins = []
for m in min:
	x = list_x[m]
	y = list_y[m]
	# print(x,y)
	m_index = max_min.index([x,y])
	prev_max = m_index-1
	next_max = prev_max + 2
	if(prev_max < 0):
		continue
	if(next_max == len(max_min)):
		continue
	heigth_th = 1
	width_th = 2
	if(y < base_line):
		continue
	
		# cv2.circle(image,(int(x),int(y)), 1, (0, 0, 255), -1)
	cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,0,255),1)
	selected_mins.append(m_index)
min_width_th = 30

# check_for_sad, dad 


# check for seen ,sheen
# for m in selected_mins:
#     next_min = m + 1
#     if(max_min[next_m] - max_min[m] < min_width_th )
# 		continue
cv2.imwrite("contoured.png",image)




