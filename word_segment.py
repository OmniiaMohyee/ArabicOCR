import cv2
import numpy as np
import copy
from preprocess import *

def line_segment(image,not_bin_image):
    im = image.copy()
    im_2 = not_bin_image.copy()
    bounds = []
    horiz_hist = np.count_nonzero(image, axis=1)
    i = 0
    while(i<len(horiz_hist)): 
        count = 0
        if(horiz_hist[i] == 0):
            j = 0
            while(i+j <len(horiz_hist)):
                if(horiz_hist[i+j]==0):
                    j+=1
                    count +=1
                else:
                    break
            if(count > 4):
                im_2[i+int(count/2),:] = 0
                bounds.append(i+int(count/2))
            i+=j
        else:
            i+=1
    line = 1
    result_lines = []
    result_lines_not_bin = []
    for i in range(len(bounds)-1):
        result_lines.append(im[bounds[i]+1:bounds[i+1],:])
        
        result_lines_not_bin.append(im_2[bounds[i]+1:bounds[i+1],:])
        # cv2.imwrite("line"+str(line)+".png",im_2[bounds[i]+1:bounds[i+1],:])
        line+=1
    return result_lines,result_lines_not_bin,line-1

def word_segment(image,not_bin_image,threshold,scale):
    (ys , xs )= image.shape
    img = cv2.resize(image,(xs*scale,ys*scale), interpolation=cv2.INTER_AREA)
    not_bin_image = cv2.resize(not_bin_image, (xs*scale, ys*scale), interpolation=cv2.INTER_AREA)
    im = img.copy()
    im_2 = not_bin_image.copy()
    vert_hist = np.count_nonzero(im > 127, axis=0)
    
    bounds = []
    k = 0
    while(k<len(vert_hist)): 
        count = 1
        if(vert_hist[k] == 0):
            j = 1
            count = 1
            while(k+j <len(vert_hist)):
                if(vert_hist[k+j]== 0):
                    j+=1
                    count +=1
                else:
                    
                    break
            if(count > threshold):
                im[:,k+int(count/2)] = 255
                im_2[:,k+int(count/2)] = 0
                bounds.append(k+int(count/2))
            k+=j
        else:
            k+=1
    word = 1
    result_words = []
    for i in range(len(bounds)-1):
        result = im[:,bounds[i]+1:bounds[i+1]]
        result = im_2[:,bounds[i]+1:bounds[i+1]]
        result_words.append(result)
        word += 1

    return result_words,word-1



def word_seg(clean_img,clean_img_not_bin):
    lines, not_bin_lines, size_lines = line_segment(clean_img,clean_img_not_bin)
    all_words = []
    count  = 0 
    for j in range(size_lines):
        words,size_words = word_segment(lines[j],not_bin_lines[j],10,5)# 1--> threshold : hyperparamter
        all_words += words
    return all_words
