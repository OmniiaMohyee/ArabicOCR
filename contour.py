import cv2
import numpy as np
from scipy.signal import argrelextrema
import copy
from PIL import Image
from generalized_hough_demo import hough_match
## SKEW DETECTION.
#1- Binarizing the image.
im_name = "c1.png"
image = cv2.imread(im_name)

def char_test(deleted_indices,segmentation_points,image,character,thr):
	# print(type(image))
	character = cv2.cvtColor(character, cv2.COLOR_RGB2GRAY)	
	# sklearn.metrics.pairwise.cosine_similarity(character,, dense_output=True)
	num_points = len(segmentation_points)
	seen_cnt,_h= cv2.findContours(character, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# print(len(seen_cnt))
	areas = [cv2.contourArea(x) for x in seen_cnt]
	m = np.argmax(areas)
	seen_cnt=seen_cnt[m]
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
		# sklearn.metrics.pairwise.cosine_similarity(character,section, dense_output=True)
		cont,_h= cv2.findContours(section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		areas = [cv2.contourArea(x) for x in cont]
		m = np.argmax(areas)
		cont=cont[m]
		hull2 = cv2.convexHull(cont)
		match = cv2.matchShapes(seen_cnt,cont,1,0.0)
		# print(hull1,hull2)
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
	# m = np.sum(normalized)
	# normalized = [float(i)/10 for i in normalized]
	for i in range(len(matches)):
		matches[i][0] = normalized[i]
	matches.sort(key =lambda x:x[0])
	normalized_matches = matches
	print(normalized_matches)

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

def hough_test(im,image,character):

	rstart,rend,cstart,cend ,max_v = hough_match(image,character)
	# print(max_v)
	# im[rstart:rend, cstart] = (255,0,0)
	# im[rstart:rend, cend] = (255,0,0)
	# im[rstart, cstart:cend] = (255,0,0)
	# im[rend, cstart:cend] = (255,0,0)

def segment(image, words_iter):
	(ys , xs , _)= image.shape
	image = cv2.resize(image,(xs*5,ys*5), interpolation=cv2.INTER_AREA)
	cv2.imwrite("cnt/resized.png", image)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	
	# for i in range(ys):
	# 	for j in range(xs) :
	# 		if gray[i][j] > 250:
	# 			gray[i][j] = 255
	# 		else:
	# 			gray[i][j] =- 0
	cv2.imwrite("cnt/gray.png",gray)
	thresh = cv2.threshold(gray, 135, 255,cv2.THRESH_BINARY)[1]
	thresh2 = cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY)[1]

	thresh_unaltered = copy.copy(thresh)
	# thresh = gray	
	# thresh = cv2.blur(thresh,(2,2))
	# cv2.imwrite("cnt/thresh.png", thresh)
	# cv2.imwrite("cnt/bin.png", bi)
	# thresh =bi
	# base line
	# thresh[line_index,:] = 255
	cv2.imwrite("cnt/bl_line.png",thresh)

	# edges 

	edges = cv2.Canny(thresh,0,500)
	cv2.imwrite("cnt/edges.png",edges)

	# countour 

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	areas = [cv2.contourArea(x) for x in contours]
	m = np.max(areas)
	normalized_areas= [float(i)/m for i in areas]
	i = np.argmax(areas)
	chars = []
	n = len(contours)
	mask = np.zeros(thresh.shape, np.uint8)
	for it in range(n):
		t=it
		cnt = contours[t]
		cv2.drawContours(image, [cnt], 0, (0,255,0), 1)
		# cv2.imwrite("cnt/contoured.png",image)
		# print(hierarchy[0,t,3],normalized_areas[t])
		if(hierarchy[0,t,3] != -1 ):
			continue
		elif normalized_areas[t] < 0.1 :
			cv2.drawContours(thresh, [cnt], 0, (0,0,0,0), -1)
			continue
		segmentation_points = []
		epsilon = 0.1*cv2.arcLength(cnt,True)
		approx = cv2.approxPolyDP(cnt,epsilon,True)

		h_line_hist = np.count_nonzero(thresh,axis=1)
		v_line_hist = np.count_nonzero(thresh,axis=0)

		line_index = np.argmax(h_line_hist)
		max = np.amax(h_line_hist)


		leftmost =tuple( cnt[cnt[:,:,0].argmin()][0])
		rightmost =tuple (cnt[cnt[:,:,0].argmax()][0])
		topmost = tuple( cnt[cnt[:,:,1].argmin()][0])
		bottommost =tuple(cnt[cnt[:,:,1].argmax()][0])


		image[rightmost[1],rightmost[0]] = (255,0,255)
		image[leftmost[1],leftmost[0]] = (255,0,255)
		image[topmost[1],topmost[0]] = (255,0,255)
		image[bottommost[1],bottommost[0]] = (255,0,255)

		threshold = 3
		base_line = line_index 
		image[base_line,:]=(0,255,255)
		list = []
		for i in range(0,len(cnt)):
			x = cnt[i][0][0]
			y = cnt[i][0][1]
			list.append((x,y))

		right = list.index(rightmost)
		left = list.index(leftmost)
		top = list.index(topmost)
		bottom = list.index(bottommost)
		l=[]
		lower_contour = []
		splitting_index = 0
		k=0
		right_reached =False 
		left_reached = False
		for i in range(0,len(cnt)):
			x = cnt[i][0][0]
			y = cnt[i][0][1]
			if x == rightmost[0]:
				right_reached = True
			if(left_reached and not right_reached and y >= base_line):
				continue
			if x == leftmost[0]:
				left_reached = True
				splitting_index = k
			if y <= base_line and [x,y] not in l:
				k+=1
				l.append([x,y])
		list =l[splitting_index+1:len(l)] + l[0:splitting_index] 
		# cv2.drawContours(image, [np.asarray(l[splitting_index:len(l)])], 0, (0,255,0), 1)
		# cv2.drawContours(image,[np.asarray(l[0:splitting_index])], 0, (0,255,0), 1)
		# cv2.drawContours(image,[cnt], 0, (0,255,0), 1)
		# cv2.imwrite("cnt/contoured.png",image)

		# if(j == 1 ):

		list=list[::-1]

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
		maximas = (np.diff(np.sign(np.diff(np.asarray(list_y)))) > 0).nonzero()[0] + 1 # local min
		minimas = (np.diff(np.sign(np.diff(np.asarray(list_y)))) < 0).nonzero()[0] + 1 # local min

		min_list = []
		max_list = []

		min_threshold = 30
		for m in minimas:
			x,y=[list_x[m],list_y[m]]
			if(y < base_line-min_threshold):
				list.remove([x,y])
				continue
			# cv2.circle(image,(int(x),int(y)), 1, (255, 0, 0), -1)
			min_list.append([x,y])
			
		threshold = 0
		image[base_line-threshold]=(255,255,0)
		
		for m in maximas:
			x,y=[list_x[m],list_y[m]]
			blue = image[y][x][0]
			green =image[y][x][1]
			red = image[y][x][2]
			color = gray[y][x]
			# print(color)
			if(y < (base_line-threshold)):
			# if(y < (base_line-threshold)):
				max_list.append([x,y])
		min_max=[]
		for i in range(len(list)):
			list_element=list[i]
			try :
				max_value = max_list.index(list_element)
				min_max.append(list_element)
				cv2.circle(image,(int(list_element[0]),int(list_element[1])),2, (255, 0, 0), -1)
			except:
				pass
			try:
				min_value = min_list.index(list_element)
				min_max.append(list_element)
				cv2.circle(image,(int(list_element[0]),int(list_element[1])),2, (0, 0, 255), -1)

			except:
				pass
		# cv2.imwrite("cnt/contoured.png",image)
		# if( len(min_list) == 0 or len(max_list ) == 0):
		# 	continue
		# temp_y = [row[1] for row in max_list]
		max_hight = 200
		splitting_points =[]
		x,y = leftmost
		splitting_points.append([x,y])
		i = 1
		avg_char_width_mid = 100
		avg_char_width_end = 110
		avg_char_area = 300

		for m in range(len(min_list)):
			[x,y] = min_list[m]
			index = min_max.index([x,y])
			prev = min_max[index-1]
			i+=1
			if prev in max_list  :		
				splitting_points.append([x,y])
		

		prev_point = leftmost
		prev_prev= leftmost
		x,y =leftmost
		segmentation_points.append([x,y])
		if([x,y] not in min_list):
			min_list = [[x,y]] + min_list
		x,y = rightmost
		splitting_points.append([x,y])
		deleted_indices = []


		min_max.append([x,y])
		if([x,y] not in min_list):
			min_list.append([x,y])


		seen=cv2.imread('cnt/seen.png')
		seen2=cv2.imread('cnt/seen2.png')
		e = cv2.Canny(seen2,0,500)
		hough_test(image,edges,e)

		for s in splitting_points :
			x = s[0]
			y = s[1]
			if((x,y) == leftmost):
				continue

			diff_min = x - prev_point[0]
			index = min_max.index([x,y])
			prev_max = min_max[index -1]
			next_max = -1
			for k in range(index+1,len(min_max)):
				if(min_max[k] in max_list):
					next_max = min_max[k]
					break
			diff_max = avg_char_width_mid

			if(next_max != -1):
				diff_max = next_max[0] - prev_max[0]	
			area = (x - prev_max[0])*(y- prev_max[1])
			print(x - prev_max[0],y- prev_max[1])
			print(area)
			if(area < 800):
				continue
			step = 5
			if(abs(s[0]-prev_max[0]) < step):
				continue
			prev_hight =base_line-prev_max[1] 
			# print(diff_min)
			# print(prev_hight , max_hight)
			# print(s, rightmost )
			if (prev_point[0] == leftmost[0] or x == rightmost[0]) and prev_hight >= max_hight and diff_min >= avg_char_width_mid/3 :
				# cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(255,0,0),2)
				segmentation_points.append([x,y])
				prev_prev = prev_point
				prev_point = s
			elif((diff_min >= avg_char_width_mid) and prev_point[0] != leftmost[0]):
    				# cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,0,255),2)
				segmentation_points.append([x,y])
				prev_prev = prev_point
				prev_point = s
			elif (diff_min >= avg_char_width_end and prev_point[0] == leftmost[0]):
				segmentation_points.append([x,y])
				prev_prev = prev_point
				prev_point = s    				
			else:
				# cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,255,255),2)
				if(prev_point in segmentation_points):
					segmentation_points.remove(prev_point)
				prev_point = s
			
		prev_point = leftmost
		num_points = len(segmentation_points)


		# print(segmentation_points)
		try:
			x,y =leftmost
			segmentation_points.index([x,y])
		except:
			segmentation_points = [[x,y]] + segmentation_points[0:num_points]
			num_points+=1
		# print(segmentation_points)
			
		try:
			x,y =rightmost
			segmentation_points.index([x,y])
		except:
			segmentation_points.append([x,y])
			num_points+=1
		# print(segmentation_points)		
		num_points = len(segmentation_points)
		for i in range(1,num_points):

			segment = segmentation_points[i]
			x = segment[0]
			y = segment[1]
			shift = 5
			if prev_point[0]!= segment[0]:
				cv2.line(image,(int(x)+shift,int(y)-50),(int(x)+shift,int(y)+50),(0,0,255),1)
				char = thresh2[:,prev_point[0]:segment[0]+shift]
				chars.append([x,char])
			prev_point = segment
	# print(len(chars))
	# xs = [row[0] for row in chars]
	# print(xs)
	chars.sort(key= lambda x :x[0])
	chars = chars[::-1]
	chars = [row[1] for row in chars]
	i =1
	for c in chars:
		cv2.imwrite("cnt/char_"+str(i)+".png",c)
		i+=1
	cv2.imwrite("cnt/contoured."+str(words_iter)+".png",image)  
	cv2.imwrite("cnt/thresh.png",thresh)

	return chars
# cs = segment(image)
