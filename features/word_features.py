import numpy as np
import math
import cv2 

#read imagen
img = cv2.imread('test.png')
img_width, img_height, channels = img.shape
_, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY) #convertt to binary

#create sliding window
wind_wdt = 3 #may be a tuning parameter
wind_hgt = int(img_height / 8) # for first level of features
img_wdt_iter = 0
wind_hgt_iter = 0

winds_count = math.ceil(img_width/3)
F = np.zeros((16,winds_count))

while img_wdt_iter < winds_count:
	while wind_hgt_iter < 8:  # to get first level of features
		wind = bw_img[img_wdt_iter*wind_wdt:(img_wdt_iter+1)*wind_wdt, wind_hgt_iter*wind_hgt:(wind_hgt_iter+1) *wind_hgt, 0]
		F[wind_hgt_iter][img_wdt_iter] = np.sum(wind == 0)
		wind_hgt_iter += 1
	i=0
	while wind_hgt_iter < 12:  # 2nd level of features
		F[wind_hgt_iter][img_wdt_iter] = F[i][img_wdt_iter] +  F[i+1][img_wdt_iter]
		wind_hgt_iter += 1
		i += 2
	while wind_hgt_iter < 15:  # 3rd level of features
		F[wind_hgt_iter][img_wdt_iter] = F[i][img_wdt_iter] +  F[i+1][img_wdt_iter] 
		wind_hgt_iter += 1
		i += 1  #F14 & F15 are opposite of in the paper, so here we add F[12] & F[14]
	F[wind_hgt_iter][img_wdt_iter] = F[12][img_wdt_iter] +  F[14][img_wdt_iter]  # 4th level of features
	img_wdt_iter += 1
	wind_hgt_iter = 0 #reset window hight iterator
