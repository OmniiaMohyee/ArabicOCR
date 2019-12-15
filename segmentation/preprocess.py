


import cv2
import numpy as np
import copy
 
def read_image(image_name):
    image = cv2.imread(image_name)
    return image

def binarize(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_gray = cv2.bitwise_not(image_gray)
    image_thresholded = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    return image_thresholded

#TODO: Add some function for noise removal.

def fix_skew(image):
    coords = np.column_stack(np.where(image > 0))
    angle  = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = - (90 + angle)
    else:
        angle = - angle
    
    (h,w) = image.shape[:2]
    center = ( w//2 , h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode= cv2.BORDER_REPLICATE)
    return rotated


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

def word_segment(image):
    
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
            if(count>1):
                image[:,k+int(count/2)] = 255
                bounds.append(k+int(count/2))
            k+=j
        else:
            k+=1
    print(bounds)
    word = 1
    result_words = []
    for i in range(len(bounds)-1):
        result = im[:,bounds[i]+1:bounds[i+1]]
        print(result)
        #cv2.imwrite("word"+str(word)+".png",result)
        result_words.append(result)
        word += 1
    #cv2.imwrite("line"+str(i+1)+".png",image)
    return result_words,word-1



folder = "./images/" 

# PIPELINE : read -> binarize -> fix_skew -> segment lines -> segment words.

# for i in range(1):
#     im = read_image(folder+str(i+2)+".png")
#     binarized = binarize(im)
#     skew_fixed = fix_skew(binarized)
#     #cv2.imwrite("fixed.png",skew_fixed)
#     lines,size_lines = line_segment(skew_fixed)
#     print(lines)
#     print(size_lines)
#     for j in range(1):
#         words,size_words = word_segment(lines[j])
#         print(len(words))
#         print(size_words)
        
        

    



