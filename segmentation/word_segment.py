import cv2
import numpy as np
import copy

def line_segment(image):
    im = image.copy()
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
            im[i+int(count/2),:] = 255
            bounds.append(i+int(count/2))
            i+=j
        else:
            i+=1
    line = 1
    result_lines = []
    for i in range(len(bounds)-1):
        result = im[bounds[i]+1:bounds[i+1],:]
        #cv2.imwrite("line"+str(line)+".png",result)
        result_lines.append(result)
        line+=1
    return result_lines,line-1

def word_segment(image,threshold):
    
    im = image.copy()
    vert_hist = np.count_nonzero(im, axis=0)
    bounds = []
    k = 0
    while(k<len(vert_hist)): 
        count = 0
        if(vert_hist[k] == 0):
            j = 0
            while(k+j <len(vert_hist)):
                if(vert_hist[k+j]==0):
                    j+=1
                    count +=1
                else:
                    break
            if(count > threshold):
                image[:,k+int(count/2)] = 255
                bounds.append(k+int(count/2))
            k+=j
        else:
            k+=1
    # print(bounds)
    word = 1
    result_words = []
    for i in range(len(bounds)-1):
        result = im[:,bounds[i]+1:bounds[i+1]]
        # print(result)
        #cv2.imwrite("word"+str(word)+".png",result)
        result_words.append(result)
        word += 1
    #cv2.imwrite("line"+str(i+1)+".png",image)
    return result_words,word-1


def word_seg(clean_img)
    lines,size_lines = line_segment(clean_img)
    for j in range(size_lines):
        words,size_words = word_segment(lines[j],1)# 1--> threshold : hyperparamter
        return words,size_words