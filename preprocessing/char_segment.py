import cv2 
import numpy as np 
from scipy.signal import argrelextrema
import copy
from PIL import Image


def hough(image):
    edges = cv2.Canny(image,50,500,apetureSize = 3)
    lines = cv2.houghLinesP(edges,1,np.pi/180,thresh = 50, minLineLength = 5, maxLineGap = 5)
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(edges,(x1,y1),(x2,y2),(255,0,0),2) 
    cv2.imwrite("linesDetected.png",edges)
    return edges

# TODO: Add a function to estimate stroke size.

def enlarge(image,scale_x,scale_y):
    (ys , xs , _)= image.shape
    enlarged = cv2.resize(image,(xs*scale_x,ys*scale_y),interpolation=cv2.INTER_AREA)
    return enlarged

def get_contour(img):
    image = copy.copy(img)
    edges = cv2.Canny(thresh,0,500)
    
    #contour
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(x) for x in contours]
    i = np.argmax(areas)
    cnt = contours[i]
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)


    leftmost =tuple( cnt[cnt[:,:,0].argmin()][0])
    rightmost =tuple (cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple( cnt[cnt[:,:,1].argmin()][0])
    bottommost =tuple(cnt[cnt[:,:,1].argmax()][0])

    # DRAWING CONTOUR POINTS.
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
    return image


