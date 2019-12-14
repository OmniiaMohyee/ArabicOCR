
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
splitting_index = 0
# print(left)
k=0
for i in range(0,len(cnt)):
	x = cnt[i][0][0]
	y = cnt[i][0][1]
	if x == leftmost[0]:
		splitting_index = k

	if y <= base_line :
		k+=1
		l.append([x,y])
		# print(x,leftmost[0])
list =l[splitting_index+1:len(l)] + l[0:splitting_index] 
# print(splitting_index,len(l))
# cv2.drawContours(image, [np.asarray(l[0:splitting_index])], 0, (0,255,0), 1)
# cv2.drawContours(image,[np.asarray(l[splitting_index+1:len(l)])], 0, (0,255,0), 1)

cv2.imwrite("contoured.png",image)

list=list[::-1]
# cv2.drawContours(image,[np.asarray(list[0:50])], 0, (0,255,0), 1)

# print(defects)
list_x=[]
list_y=[]
# list_x = list[:,0]
# list_y = list[:,1]
# indexes = np.unique(list_x, return_index=True)[1]
# list_x = [list_x[index] for index in sorted(indexes)]
last_y = 0

l=copy.copy(list)
for i in range(len(l)):
	y= l[i][1]
	x= l[i][0]
	if(y != last_y ):
		list_x.append(x)
		list_y.append(y)
	else:
		list.remove([x,y])
	last_y = y
	# cv2.circle(image,(int(x),int(y)), 1, (0, 0, 255), -1)

# minimas = argrelextrema(np.asarray(list_x), np.less)
# print(list_y)
maximas = (np.diff(np.sign(np.diff(np.asarray(list_y)))) > 0).nonzero()[0] + 1 # local min
minimas = (np.diff(np.sign(np.diff(np.asarray(list_y)))) < 0).nonzero()[0] + 1 # local min
# print([[list_x[i],list_y[i]] for i in maximas] )
# print([[list_x[i],list_y[i]] for i in minimas] )

min_list = []
max_list = []


for m in minimas:
	x,y=[list_x[m],list_y[m]]
	# print(base_line,y)
	if(y < base_line):
		list.remove([x,y])
		continue
	cv2.circle(image,(int(x),int(y)), 1, (255, 0, 0), -1)
	min_list.append([x,y])

for i in range(len(list)) :
	x = list[i][0]
	y = list[i][1]
	# cv2.circle(image,(int(x),int(y)),1, (0, 255,255), -1)
    

for m in maximas:
	x,y=[list_x[m],list_y[m]]
	max_list.append([x,y])
	cv2.circle(image,(int(x),int(y)),1, (255, 255, 0), -1)
min_max=[]
for i in range(len(list)):
	list_element=list[i]
	try :
		max_value = max_list.index(list_element)
		min_max.append(list_element)
	except:
		pass
	try:
		min_value = min_list.index(list_element)
		min_max.append(list_element)
	except:
		pass
print(min_max)
print(max_list)
print(min_list)
# max_min.sort(key=lambda x:x[0])
# print(min_max)
cv2.imwrite("contoured.png",image)

selected_mins = []
splitting_points =[]
char_width = []
m_pairs =[]
for m in range(len(min_list)):
	[x,y] = min_list[m]
	# print([x,y])
	index = min_max.index([x,y])
	prev = min_max[index-1]
	# cv2.circle(image,(prev[0],prev[1]), 1, (0, 0, 255), -1)
 
	# print(max)
	# print(prev)
	if prev in max_list :		
		splitting_points.append([x,y])
		# cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,0,255),1)
prev_point = leftmost
avg_char_width = 30
avg_char_area = 600
segmentation_points = []
for s in splitting_points :
	x = s[0]
	y = s[1]
	diff = x - prev_point[0]
	index = min_max.index([x,y])
	prev = min_max[index -1 ]
	area = (base_line - prev[1])*(diff)
	print(area)
	# print(diff)
	if(diff >= avg_char_width and  area > avg_char_area):
		segmentation_points.append([x,y])
		cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,0,255),1)
		prev_point = s
prev_point = leftmost
for i in range(len(segmentation_points)):

	segment = segmentation_points[i]
	# index = min_max.index(segment)
	# prev = min_max[index -1 ]
	# char = image[prev[1]:bottommost[1],prev_point[0]:segment[0]]
	char = image[:,prev_point[0]:segment[0]]
	print(prev_point[0],segment[0])
	print(char)
	cv2.imwrite(str(i+1)+".png",char)
	prev_point = segment

cv2.imwrite("contoured.png",image)




