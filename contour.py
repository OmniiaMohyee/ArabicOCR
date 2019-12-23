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

def char_test_seen(deleted_indices,segmentation_points,image,character,thr):
	character = cv2.cvtColor(character, cv2.COLOR_RGB2GRAY)	
	num_points = len(segmentation_points)
	seen_cnt,_h= cv2.findContours(character, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	areas = [cv2.contourArea(x) for x in seen_cnt]
	m = np.argmax(areas)
	seen_cnt=seen_cnt[m]
	to_remove =[]
	matches = []
	for i in range(num_points):
		if(i+3 >= num_points ):
			break
		s = segmentation_points[i]
		f = segmentation_points[i+3]
		section = image[:,s[0]:f[0]]
		cont,_h= cv2.findContours(section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		areas = [cv2.contourArea(x) for x in cont]
		m = np.argmax(areas)
		cont=cont[m]
		hull2 = cv2.convexHull(cont)
		match = cv2.matchShapes(seen_cnt,cont,1,0.0)
		cv2.imwrite("cnt/"+str(i)+".png",section)
		matches.append([match,i])
	if len(matches) == 0:
		return

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
def char_test_saad(deleted_indices,segmentation_points,image,character,thr):
	character = cv2.cvtColor(character, cv2.COLOR_RGB2GRAY)	
	num_points = len(segmentation_points)
	saad_cnt,_h= cv2.findContours(character, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	areas = [cv2.contourArea(x) for x in saad_cnt]
	m = np.argmax(areas)
	saad_cnt=saad_cnt[m]
	to_remove =[]
	matches = []
	for i in range(num_points):
		if(i+2 >= num_points ):
			break
		s = segmentation_points[i]
		f = segmentation_points[i+2]
		section = image[:,s[0]:f[0]]
		cont,_h= cv2.findContours(section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		areas = [cv2.contourArea(x) for x in cont]
		m = np.argmax(areas)
		cont=cont[m]
		hull2 = cv2.convexHull(cont)
		match = cv2.matchShapes(saad_cnt,cont,1,0.0)
		cv2.imwrite("cnt/"+str(i)+".png",section)
		matches.append([match,i])
	if len(matches) == 0:
		return

	matches.sort(key =lambda x:x[0])
	normalized_matches = matches

	k =0 
	for k in range(len(normalized_matches)):
		index = normalized_matches[k][1]
		match = normalized_matches[k][0]
		if(match  > thr ):
			break
		if (segmentation_points[index] in deleted_indices ):
			continue
		false_line1 = segmentation_points[index+1]
		to_remove.append(false_line1)
		deleted_indices.append(segmentation_points[index])
		deleted_indices.append(segmentation_points[index+1])
		deleted_indices.append(segmentation_points[index+2])

	for r in to_remove:
		if r in segmentation_points:
			segmentation_points.remove(r)

def segment(image, words_iter):
	(ys , xs , _)= image.shape
	image = cv2.resize(image,(xs*5,ys*5), interpolation=cv2.INTER_AREA)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	
	thresh = cv2.threshold(gray, 130, 255,cv2.THRESH_BINARY)[1]
	thresh2 = cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY)[1]

	cv2.imwrite("cnt/thresh.png", thresh)

	edges = cv2.Canny(thresh,0,500)

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
		if(hierarchy[0,t,3] != -1 ):
			continue
		elif normalized_areas[t] < 0.1 or areas[t] < 5000:
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
		first_max = 0
		for i in range(0,len(cnt)):
			x = cnt[i][0][0]
			y = cnt[i][0][1]
			if x == rightmost[0]:
				right_reached = True
			if(left_reached and not right_reached):
				continue
			if x == leftmost[0]:
				first_max = [x,y]

				left_reached = True
				splitting_index = k
			if y <= base_line and [x,y] not in l:
				k+=1
				l.append([x,y])
		list =l[splitting_index+1:len(l)] + l[0:splitting_index+1] 

		list=list[::-1]

		list_x=[]
		list_y=[]
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
			min_list.append([x,y])
			
		threshold = 0
		image[base_line-threshold]=(255,255,0)
		if(first_max[1] < (base_line-threshold)):
			max_list.append([first_max[0],first_max[1]])
		for m in maximas:
			x,y=[list_x[m],list_y[m]]
			if(y < (base_line-threshold)):
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
		max_hight_left = 220
		max_hight_right = 220
		# max_hight_right = [r[1] for r in max_list]
		# try:
		# 	max_hight_right=np.min(max_hight_right)
		# 	max_hight_right = base_line -max_hight_right
		# except:
		# 	max_hight_right = max_hight_left
		splitting_points =[]
		x,y = leftmost
		splitting_points.append([x,y])
		i = 1
		avg_char_width_mid = 70
		avg_char_width_end = 110
		avg_char_area = 300

		for m in range(len(min_list)):
			[x,y] = min_list[m]
			index = min_max.index([x,y])
			if(index == 0):
				continue
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


		min_max.append([x,y])
		if([x,y] not in min_list):
			min_list.append([x,y])


		p = True
		# print(splitting_points)
		all_hights =[]
		for l in range(len(splitting_points)) :
			s = splitting_points[l]
			x = s[0]
			y = s[1]
			if((x,y) == leftmost):
				continue

			diff_min = x - prev_point[0]
			index = min_max.index([x,y])
			prev_max = min_max[index -1]
			next_max = [x,base_line]
			if((x,y) != rightmost):
				for k in range(index+1,len(min_max)):
					if(min_max[k] in max_list):
						next_max = min_max[k]
						break
				next_min = splitting_points[l+1]		
			else:
				next_min = rightmost
			area = abs(x - prev_max[0])*abs(y- prev_max[1])
			if(area < 1000):
				continue
			step = 5
			if(abs(s[0]-prev_max[0]) < step):
				continue
			prev_hight = base_line - prev_max[1] 
			next_hight = base_line - next_max[1]
			# saad check
			r1 = (next_min[0] - x)/(rightmost[1]- next_max[1])
			r2 = (x - prev_max[0])/(rightmost[1] - prev_max[1])
			# print(diff_min)
			# print(prev_hight , max_hight_left , next_hight , max_hight_right)
			# print(s, rightmost ,leftmost)
			all_hights.append(prev_hight)


			if (prev_point[0] == leftmost[0] and prev_hight >= max_hight_left ) or ( next_min[0] == rightmost[0] and next_hight >= max_hight_right and next_min[0]-x < avg_char_width_end )  and diff_min >= avg_char_width_end/3 :
				# print('0')
				# cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(255,0,0),2)
				p = False
				segmentation_points.append([x,y])
				prev_prev = prev_point
				prev_point = s
			elif((diff_min >= avg_char_width_mid) and prev_point[0] != leftmost[0]):
    				# cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,0,255),2)
				# print('1')
				p= True
				segmentation_points.append([x,y])
				prev_prev = prev_point
				prev_point = s
			elif (diff_min >= avg_char_width_end and prev_point[0] == leftmost[0]):
				# print('2')
				p = True
				segmentation_points.append([x,y])
				prev_prev = prev_point
				prev_point = s    				
			elif(p):
				# print(3)
				# cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,255,255),2)
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
			num_points+=1
			
		try:
			x,y =rightmost
			segmentation_points.index([x,y])
		except:
			segmentation_points.append([x,y])
			num_points+=1
		s_counts = 0
		s_thr = 100
		indx = -1
		to_remove = []
		# for i in range(len(segmentation_points)):
		# 	p = segmentation_points[i]
		# 	if(p[0] == leftmost[0] or p[0] ==rightmost[0]):
		# 		continue
		# 	print(p)
		# 	next_p = segmentation_points[i+1]
		# 	prev_m = min_max[min_max.index(p)-1]
		# 	next_m = min_max[min_max.index(next_p)-1]
		# 	diff = next_m[0]-prev_m[0]
		# 	print(diff)
		# 	if( diff <=s_thr):
		# 		s_counts+=1
		# 		if(s_counts == 1):
		# 			indx = i
		# 	else:
		# 		s_counts = 0
		# 		indx = -1
		# 	if(s_counts == 2):
		# 		to_remove.append(indx)
		# 		to_remove.append(indx+1)
		# for i in range(len(to_remove)):
		# 	print(i)
		# 	s = to_remove[i]
		# 	if s in segmentation_points:
		# 		segmentation_points.remove(s)
		deleted_indices = []
		deleted_indices_saad = []

		seen=cv2.imread('cnt/s2.png') # 2
		char_test_seen(deleted_indices,segmentation_points,thresh,seen,0.5)

		seen2=cv2.imread('cnt/s6.png') # 6
		char_test_seen(deleted_indices,segmentation_points,thresh,seen2,0.5)

		seen3=cv2.imread('cnt/s24.png') #  1
		char_test_seen(deleted_indices,segmentation_points,thresh,seen3,0.5)

		seen4=cv2.imread('cnt/s9.png') # 1
		char_test_seen(deleted_indices,segmentation_points,thresh,seen4,0.5)

		seen5=cv2.imread('cnt/s23.png') # 1
		char_test_seen(deleted_indices,segmentation_points,thresh,seen5,0.5)

		seen6=cv2.imread('cnt/s35.png') # 1
		char_test_seen(deleted_indices,segmentation_points,thresh,seen6,0.5)

		seen7=cv2.imread('cnt/s36.png') # 1
		char_test_seen(deleted_indices,segmentation_points,thresh,seen7,0.5)

		daad = cv2.imread("cnt/d1.png")
		char_test_saad(deleted_indices_saad,segmentation_points,thresh,daad,0.5)
		daad2 = cv2.imread("cnt/d2.png")
		char_test_saad(deleted_indices_saad,segmentation_points,thresh,daad2,0.2)

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
	chars.sort(key= lambda x :x[0])
	chars = chars[::-1]
	chars = [row[1] for row in chars]
	# cv2.imwrite("cnt/contoured."+str(words_iter)+".png",image)  

	return chars
# cs = segment(image)
