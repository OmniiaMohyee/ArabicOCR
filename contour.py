import cv2
import numpy as np
from scipy.signal import argrelextrema
import copy
from PIL import Image
## SKEW DETECTION.
#1- Binarizing the image.
im_name = "c1.png"
image = cv2.imread(im_name)

# def hist_test(segmentation_points,image,character):
# 	cv2.cvtColor(character, cv2.COLOR_BGR2HSV)
# 	( cropy , cropx , _)= character.shape
# 	print(character.shape)
# 	h_bins = 50
# 	s_bins = 60
# 	histSize = [h_bins, s_bins]
# 	# hue varies from 0 to 179, saturation from 0 to 255
# 	h_ranges = [0, 180]
# 	s_ranges = [0, 256]
# 	ranges = h_ranges + s_ranges # concat lists
# 	# Use the 0-th and 1-st channels
# 	channels = [0, 1]
# 	hist1= cv2.calcHist([character], channels, None, histSize, ranges, accumulate=False)
# 	cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# 	hist1 = cv2.calcHist([character],[0],None,[256],[0,256])
# 	num_points = len(segmentation_points)
# 	hist_result = []
# 	for i in range(num_points):
# 		if(i+3) >= num_points:
# 			break
# 		s = segmentation_points[i]
# 		f = segmentation_points[i+3]
# 		section = image[:,s[0]:f[0]]
# 		cv2.cvtColor(section, cv2.COLOR_BGR2HSV)
# 		(y,x,_) = section.shape
# 		startx = x//2-(cropx//2)
# 		starty = y//2-(cropy//2)    
# 		section =section[starty:starty+cropy,startx:startx+cropx]
# 		print(section.shape)
# 		hist2= cv2.calcHist([section], channels, None, histSize, ranges, accumulate=False)
# 		cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
# 		res = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
# 		hist_result.append(res)
# 	print(hist_result)
def char_test(deleted_indices,segmentation_points,image,character,thr):
	# print(type(image))
	character = cv2.cvtColor(character, cv2.COLOR_RGB2GRAY)	
	num_points = len(segmentation_points)
	seen_cnt,_h= cv2.findContours(character, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	seen_cnt=seen_cnt[0]
	hull1 = cv2.convexHull(seen_cnt)
	# print(num_points)
	to_remove =[]
	# print(segmentation_points)
	matches = []
	for i in range(num_points):
		# print(i)
		if(i+3 >= num_points ):
			break
		s = segmentation_points[i]
		f = segmentation_points[i+3]
		section = image[:,s[0]:f[0]]
		cont,_h= cv2.findContours(section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		areas = [cv2.contourArea(x) for x in cont]
		m = np.argmax(areas)
		cont=cont[m]
		hull2 = cv2.convexHull(cont)
		match = cv2.matchShapes(seen_cnt,cont,1,0.0)
		# print(hull1,hull2)
		# cv2.drawContours(image, [cont], 0, (0,255,0), 1)
		cv2.imwrite(str(i)+".png",section)
		# print(match)
		# print(i)
		matches.append([match,i])
	# print(matches)
	# matches.sort(key = lambda x:x[0])
	if len(matches) == 0:
		return
	normalized = [row[0] for row in matches]
	# print("normalized", normalized)
	m = np.max(normalized)
	normalized = [float(i)/m for i in normalized]
	for i in range(len(matches)):
		matches[i][0] = normalized[i]
	matches.sort(key =lambda x:x[0])
	normalized_matches = matches
	# print(normalized_matches)
	k =0 
	for k in range(len(normalized_matches)):
		index = normalized_matches[k][1]
		match = normalized_matches[k][0]
		if(match  > thr ):
			break
		if (segmentation_points[index] in deleted_indices ):
			continue
		false_line1 = segmentation_points[index+1]
		false_line2 = segmentation_points[index+2]
		to_remove.append(false_line1)
		to_remove.append(false_line2)
		deleted_indices.append(segmentation_points[index])
		deleted_indices.append(segmentation_points[index+1])
		deleted_indices.append(segmentation_points[index+2])
		deleted_indices.append(segmentation_points[index+3])

	for r in to_remove:
		if r in segmentation_points:
			segmentation_points.remove(r)
	# print(normalized_matches)

def segment(image, words_iter):
	(ys , xs , _)= image.shape
	image = cv2.resize(image,(xs*5,ys*5), interpolation=cv2.INTER_AREA)
	cv2.imwrite("resized.png", image)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	
	# for i in range(ys):
	# 	for j in range(xs) :
	# 		if gray[i][j] > 250:
	# 			gray[i][j] = 255
	# 		else:
	# 			gray[i][j] = 0
	cv2.imwrite("gray.png",gray)
	thresh = cv2.threshold(gray, 250, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# thresh = gray	
	cv2.imwrite("thresh.png",thresh)
	# thresh = cv2.blur(thresh,(2,2))
	# cv2.imwrite("thresh.png", thresh)
	# cv2.imwrite("bin.png", bi)
	# thresh =bi
	# base line
	h_line_hist = np.count_nonzero(thresh,axis=1)
	v_line_hist = np.count_nonzero(thresh,axis=0)

	line_index = np.argmax(h_line_hist)
	max = np.amax(h_line_hist)
	# thresh[line_index,:] = 255
	cv2.imwrite("bl_line.png",thresh)

	# edges 

	edges = cv2.Canny(thresh,0,500)
	cv2.imwrite("edges.png",edges)

	# countour 

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
	areas = [cv2.contourArea(x) for x in contours]
	i = np.argmax(areas)
	chars = []
	ordered =[]
	segmentation_points = []
	cnt = contours[i]
	epsilon = 0.1*cv2.arcLength(cnt,True)
	approx = cv2.approxPolyDP(cnt,epsilon,True)
	# print(cnt)

	leftmost =tuple( cnt[cnt[:,:,0].argmin()][0])
	rightmost =tuple (cnt[cnt[:,:,0].argmax()][0])
	topmost = tuple( cnt[cnt[:,:,1].argmin()][0])
	bottommost =tuple(cnt[cnt[:,:,1].argmax()][0])


	# image[rightmost[1],rightmost[0]] = (255,0,0)
	# image[leftmost[1],leftmost[0]] = (255,0,0)
	# image[topmost[1],topmost[0]] = (255,0,0)
	# image[bottommost[1],bottommost[0]] = (255,0,0)

	threshold = 3
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
	lower_contour = []
	splitting_index = 0
	# print(left)
	k=0
	right_reached =False 
	for i in range(0,len(cnt)):
		x = cnt[i][0][0]
		y = cnt[i][0][1]
		if x == rightmost[0]:
			right_reached = True
		if(splitting_index != 0 and not right_reached):
			lower_contour.append([x,y])
			continue
		if x == leftmost[0]:
			splitting_index = k
		if y <= base_line and [x,y] not in l:
			k+=1
			l.append([x,y])
			# print(x,leftmost[0])
	list =l[splitting_index+1:len(l)] + l[0:splitting_index] 
	# print(splitting_index,len(l))
	# cv2.drawContours(image, [np.asarray(l[splitting_index+1:len(l)])], 0, (0,255,0), 1)
	# cv2.drawContours(image,[np.asarray(l[0:splitting_index])], 0, (0,255,0), 1)
	# if(j == 1 ):
	# cv2.drawContours(image, [cnt], 0, (0,255,0), 1)

	cv2.imwrite("contoured.png",image)

	list=list[::-1]
	# cv2.drawContours(image,[cnt], 0, (0,255,0), 1)

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
		# cv2.circle(image,(int(x),int(y)), 1, (255, 0, 0), -1)
		min_list.append([x,y])
		
	threshold = 24
	for m in maximas:
		x,y=[list_x[m],list_y[m]]
		blue = image[y][x][0]
		green =image[y][x][1]
		red = image[y][x][2]
		# if(y < (base_line-threshold) and blue <= 100 and green <= 100 and red <= 100):
		if(y < (base_line-threshold)):
			max_list.append([x,y])
			# cv2.circle(image,(int(x),int(y)),1, (255, 255, 0), -1)
	min_max=[]
	for i in range(len(list)):
		list_element=list[i]
		try :
			max_value = max_list.index(list_element)
			min_max.append(list_element)
			cv2.circle(image,(int(list_element[0]),int(list_element[1])),1, (255, 0, 0), -1)
		except:
			pass
		try:
			min_value = min_list.index(list_element)
			min_max.append(list_element)
			cv2.circle(image,(int(list_element[0]),int(list_element[1])),1, (0, 0, 255), -1)

		except:
			pass
	# print(min_max)
	# print(max_list)
	# print(min_list)
	# max_min.sort(key=lambda x:x[0])
	# print(min_max)
	cv2.imwrite("contoured.png",image)

	selected_mins = []
	splitting_points =[]
	char_width = []
	m_pairs =[]
	i = 1
	for m in range(len(min_list)):
		[x,y] = min_list[m]
		index = min_max.index([x,y])
		prev = min_max[index-1]
		i+=1
		if prev in max_list :		
			# cv2.circle(image,(prev[0],prev[1]), 1, (0, 0, 255), -1)
			# cv2.circle(image,(x,y), 1, (0, 255, 0), -1)
			splitting_points.append([x,y])
			# cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,0,255),1)
	
	# solve conflicts in lower areas
	# image[base_line,:]=(255,0,0)
	# lower_y = [row[1] for row in lower_contour]
	# lower_maximas = (np.diff(np.sign(np.diff(np.asarray(lower_y)))) > 0).nonzero()[0] + 1 # local min
	# lower_minimas = (np.diff(np.sign(np.diff(np.asarray(lower_y)))) < 0).nonzero()[0] + 1 # local min

	# for m in lower_minimas:
	# 	x,y = lower_contour[m]
	# 	if( y > base_line):
	# 		cv2.circle(image,(x,y), 1, (0, 255, 0), -1)

	# for m in lower_maximas:
	# 	x,y = lower_contour[m]
	# 	if( y > base_line):
	# 		cv2.circle(image,(x,y), 1, (0, 0, 255), -1)


	######################3

	prev_point = leftmost
	prev_prev= leftmost
	avg_char_width = 14
	avg_char_area = 300
	x,y =leftmost
	segmentation_points.append([x,y])
	# print(segmentation_points)
	x,y =rightmost
	splitting_points.append([x,y])
	min_max.append([x,y])
	for s in splitting_points :
		x = s[0]
		y = s[1]
		diff_min = x - prev_point[0]
		index = min_max.index([x,y])
		prev_max = min_max[index -1]
		next_max = -1
		for k in range(index+1,len(min_max)):
			# print(min_max[k])
			# print(max_list)
			if(min_max[k] in max_list):
				next_max = min_max[k]
				break
		diff_max = avg_char_width

		if(next_max != -1):
			diff_max = next_max[0] - prev_max[0]	
			# print(diff)
		area = (base_line - prev_max[1])*(diff_max)
		step = 5
		if(abs(s[0]-prev_max[0]) < 5):
			continue
		# print(diff_min)
		if(diff_min >= avg_char_width ):
			# cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,0,255),1)
			segmentation_points.append([x,y])
			prev_prev = prev_point
			prev_point = s
		else:
			cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,255,255),1)
			if(prev_point in segmentation_points):
				segmentation_points.remove(prev_point)
			prev_point = s
	prev_point = leftmost
	num_points = len(segmentation_points)
	try:
		x,y =leftmost
		segmentation_points.index([x,y])
	except:
		segmentation_points = [[x,y]] + segmentation_points[0:num_points]
		# print(segmentation_points)
		num_points+=1
	# print(leftmost)
	# print(segmentation_points)
	try:
		x,y =rightmost
		segmentation_points.index([x,y])
	except:
		segmentation_points.append([x,y])
		# print(rightmost)
		# print(segmentation_points)
		num_points+=1
	# print(segmentation_points)
	
	deleted_indices = []
	# # sheen
	# print(segmentation_points)
	seen2=cv2.imread('seen2.png')
	# char_test(deleted_indices,segmentation_points,thresh,seen2,0.1)

	if(len(deleted_indices) == 0):
		seen =cv2.imread('seen.png')
		# char_test(deleted_indices,segmentation_points,thresh,seen,0.1)


	# seen3=cv2.imread('seen3.png')
	# char_test(deleted_indices,segmentation_points,thresh,seen3,0.2)

	num_points = len(segmentation_points)
	# print(num_points)
	for i in range(1,num_points):

		# index = min_max.index(segment)
		# prev = min_max[index -1 ]
		# char = image[prev[1]:bottommost[1],prev_point[0]:segment[0]]
		# if(i == num_points-1 ):
		# 	print('last point')
		# 	segmentation_points[i][0]=rightmost[0]
		# 	segmentation_points[i][1]=rightmost[1]
		segment = segmentation_points[i]
		# print(segment)
		x = segment[0]
		y = segment[1]
		shift = 5
		if prev_point[0]!= segment[0]:
			cv2.line(image,(int(x)+shift,int(y)-50),(int(x)+shift,int(y)+50),(0,0,255),1)
			char = thresh[:,prev_point[0]:segment[0]+shift]
			chars.append([x,char])
		# print(prev_point[0],segment[0])
		# print(char)
		prev_point = segment
	# xs = [row[0] for row in chars]
	# print(xs)
	chars.sort(key= lambda x :x[0])
	chars = chars[::-1]
	chars = [row[1] for row in chars]
	i =1
	for c in chars:
		# cv2.imwrite("char_"+str(i)+".png",c)
		i+=1
	# cv2.imwrite("contoured."+str(words_iter)+".png",image)  
	return chars
# cs = segment(image)
