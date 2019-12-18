import cv2
import numpy as np


def contours_and_cetroid(thresh,img):
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0,255,0), 3)
    # for i in range (len(contours)): #just to draw them with different colors
    #     if i ==  0:
    #         cv2.drawContours(img, contours, i, (255, 125, 0), 3) 
    #         cv2.imshow('Contours', img) 
    #         cv2.waitKey(0) 
    #     elif i ==  1:
    #         cv2.drawContours(img, contours, i, (125, 255, 0), 3) 
    #         cv2.imshow('Contours', img) 
    #         cv2.waitKey(0) 

    #get char centroid
    cnt = contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return len(contours), (cx, cy)


def build_R_Table(img,centroid): #binary image, character center
    table = [[0 for x in range(1)] for y in range(90)]  # creating a empty list
    x2, y2 = centroid # r will be calculated w.r.t this point
    for x1 in range(0, img.shape[0]):
        for y1 in range(0, img.shape[1]):
            if img[x1, y1] != 0:      
                r = [(x2-x1), (y2-y1)]
                if (x2-x1 != 0):  
                    theta = np.arctan(int((y2-y1)/(x2-x1))) #theta = arctan(delta_y / delta_x)
                    theta = int(np.rad2deg(theta)) 
                    table[np.absolute(theta)].append(r)

    for i in range(len(table)):
        table[i].pop(0)
    return table


def main():
    #read image and convert it to binary 
    img = cv2.imread("mem.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray) #find countours need character to be white and background black
    thresh = cv2.threshold(gray, 127,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    num_contours, centroid= contours_and_cetroid(thresh, img)
    print("number of contours found -> ", num_contours)

    edges = cv2.Canny(gray,50,200,apertureSize = 3)
    R_table = build_R_Table(edges,centroid)
   
    img[:,[cx]] = (255,0,0)
    # img[cy,:] = (255,0,0)
    cv2.imshow('Contours', img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 


main()


def matchTable(img, table):
    m, n = img.shape
    acc = np.zeros((m+50, n+50))  # acc array requires some extra space

    def findGradient(x, y):
        if (x != 0):
            return int(np.rad2deg(np.arctan(int(y/x))))
        else:
            return 0

    for x in range(1, img.shape[0]):
        for y in range(img.shape[1]):

            if img[x, y] != 0:  # boundary point
                theta = findGradient(x, y)
                vectors = table[theta]
                for vector in vectors:
                    acc[vector[0]+x, vector[1]+y] += 1
    return acc

def findMaxima(acc):
    """
    :param acc: accumulator array
    :return:
        maxval: maximum value found
        ridx: row index of the maxval
        cidx: column index of the maxval
    """
    ridx, cidx = np.unravel_index(acc.argmax(), acc.shape)
    return [acc[ridx, cidx], ridx, cidx]