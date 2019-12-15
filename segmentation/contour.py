import cv2
import numpy as np
from scipy.signal import argrelextrema
import copy
from PIL import Image
## SKEW DETECTION.
#1- Binarizing the image.
im_name = "madrasa.png"
image = cv2.imread(im_name)


def segment(image):
	(ys , xs , _)= image.shape
	image = cv2.resize(image,(xs*5,ys*5), interpolation=cv2.INTER_AREA)
	cv2.imwrite("resized.png", image)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
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

	c1_edges = cv2.Canny(thresh,0,500)
	cv2.imwrite("edged.png",c1_edges)

	# countour 

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
	areas = [cv2.contourArea(x) for x in contours]
	i = np.argmax(areas)
	for j in (range(len(contours))):
		cnt = contours[j]
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

			if y <= base_line and [x,y] not in l:
				k+=1
				l.append([x,y])
				# print(x,leftmost[0])
		list =l[splitting_index+1:len(l)] + l[0:splitting_index] 
		# print(splitting_index,len(l))
		# cv2.drawContours(image, [np.asarray(l[splitting_index+1:len(l)])], 0, (0,255,0), 1)
		# cv2.drawContours(image,[np.asarray(l[0:splitting_index])], 0, (0,255,0), 1)
		# cv2.drawContours(image, [np.asarray(l)], 0, (0,255,0), 1)

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
			# cv2.circle(image,(int(x),int(y)), 1, (255, 0, 0), -1)
			min_list.append([x,y])
			
		threshold = 10
		for m in maximas:
			x,y=[list_x[m],list_y[m]]
			if(y < base_line - threshold):
				max_list.append([x,y])
				cv2.circle(image,(int(x),int(y)),1, (255, 255, 0), -1)
		min_max=[]
		for i in range(len(list)):
			list_element=list[i]
			try :
				max_value = max_list.index(list_element)
				min_max.append(list_element)
				# cv2.circle(image,(int(list_element[0]),int(list_element[1])),1, (0, 255, 0), -1)
			except:
				pass
			try:
				min_value = min_list.index(list_element)
				min_max.append(list_element)
				# cv2.circle(image,(int(list_element[0]),int(list_element[1])),1, (0, 0, 255), -1)

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
			# print([x,y])
			index = min_max.index([x,y])
			prev = min_max[index-1]
			# cv2.circle(image,(prev[0],prev[1]), 1, (0, 0, 255), -1)
			# cv2.circle(image,(x,y), 1, (0, 255, 0), -1)
			# if(i == 3):
			# 	cv2.circle(image,(prev[0],prev[1]), 1, (0, 0, 255), -1)
			# 	cv2.circle(image,(x,y), 1, (0, 255, 0), -1)
			# print(i)
			# print(max_list)
			# print(prev)
			i+=1
			if prev in max_list :		
				# cv2.circle(image,(prev[0],prev[1]), 1, (0, 0, 255), -1)
				# cv2.circle(image,(x,y), 1, (0, 255, 0), -1)
				splitting_points.append([x,y])
				# cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,0,255),1)
		prev_point = leftmost
		avg_char_width = 20
		avg_char_area = 300
		segmentation_points = []
		for s in splitting_points :
			x = s[0]
			y = s[1]
			# diff = x - prev_point[0]
			index = min_max.index([x,y])
			prev_max = min_max[index -1]
			next_max = -1
			for k in range(index+1,len(min_max)):
				# print(min_max[k])
				# print(max_list)
				if(min_max[k] in max_list):
					next_max = min_max[k]
					break
			diff = avg_char_width

			if(next_max != -1):
				diff = next_max[0] - prev_max[0]	
				print(diff)
			area = (base_line - prev_max[1])*(diff)
			if(diff >= avg_char_width):
				# cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,0,255),1)
				segmentation_points.append([x,y])
				prev_point = s
		prev_point = leftmost
		num_points = len(segmentation_points)
		for i in range(num_points):

			# index = min_max.index(segment)
			# prev = min_max[index -1 ]
			# char = image[prev[1]:bottommost[1],prev_point[0]:segment[0]]
			if(i == num_points-1 ):
				print('last point')
				segmentation_points[i][0]=rightmost[0]
				segmentation_points[i][1]=rightmost[1]
			segment = segmentation_points[i]
			x = segment[0]
			y = segment[1]
			cv2.line(image,(int(x),int(y)-50),(int(x),int(y)+50),(0,0,255),1)
			char = image[:,prev_point[0]:segment[0]]
			# print(prev_point[0],segment[0])
			# print(char)
			cv2.imwrite(str(i+1)+".png",char)
			prev_point = segment

		cv2.imwrite("contoured.png",image)
segment(image)