import numpy as np
import math
import cv2 

# 1.exact width and height
def crop_image(img):
    img_height, img_width = img.shape
    first_row = img_height
    first_col = img_width
    last_row = 0
    last_col = 0
    for i in range (img_height):
        for j in range(img_width):
            if (img[i][j]==0):
                first_row = min(first_row,i)
                first_col = min(first_col,j)
                last_row = max (last_row,i)
                last_col = max (last_col,j)
    cropped_img = img[first_row:last_row,first_col:last_col]
    # cv2.imshow("img",cropped_img)
    return cropped_img

# 2. extract black and white pixels
def get_black_and_white(img):
    black_pixels = 0
    white_pixels = 0
    width,height = img.shape
    for i in range (width):
        for j in range(height):
            if (img[i][j]==0):
                black_pixels +=1
            else:
                white_pixels +=1
    return black_pixels,white_pixels
# 3. horizontal transitions
def horizontal_transitions(img):
    horizontal =0
    width,height = img.shape
    for i in range (width):
        for j in range(1,height):
            if (img[i][j] != img[i][j-1]):
                horizontal +=1
    return horizontal
# 4. Vertical transitions
def vertical_transitions(img):
    vertical = 0
    width,height = img.shape
    for j in range (height):
        for i in range (1,width):
            if (img[i][j] != img[i-1][j]):
                vertical +=1
    return vertical

def get_Regions(img):
    height, width = img.shape
    Region_1 = img[:width//2,:height//2]
    Region_2 = img[:width//2,height//2+1:]
    Region_3 = img[width//2+1:,:height//2]
    Region_4 = img[width//2+1:,height//2+1:]
    Regions= [Region_1,Region_2,Region_3,Region_4]
    # cv2.imshow("r1",Region_1)
    # cv2.imshow("r2",Region_2)
    # cv2.imshow("r3",Region_3)
    # cv2.imshow("r4",Region_4)
    return Regions

def getFeatureVector(cropped_img):
    FeatureVector=[]
    width,height = cropped_img.shape
    FeatureVector.append(height/width) #1. height/width
    b,w = get_black_and_white(cropped_img)
    FeatureVector.append(b/w) #2.black/white
    FeatureVector.append(horizontal_transitions(cropped_img)) # 3. horizontal transitions
    FeatureVector.append(vertical_transitions(cropped_img)) # 4. Vertical transitions

    regions = get_Regions(cropped_img)
    black=[]
    white=[]
    for i in range (len(regions)):
        b,w = get_black_and_white(regions[i])
        black.append(b)
        white.append(w)
    for i in range (4):
        if white[i] == 0:
            FeatureVector.append(1) #this 1 is just a fatea
        else:
            FeatureVector.append(black[i]/white[i])
    for i in range(3):
        for j in range(i+1,4):
            if black[j] == 0:
                FeatureVector.append(1) #this 1 is just a fatea
            else:
                FeatureVector.append(black[i]/black[j])
    return FeatureVector

# if __name__=="__main__":
    # #read image
    # img = cv2.imread('t2214.png')
    # img_grey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # _, bw_img = cv2.threshold(img_grey,127,255,cv2.THRESH_BINARY) #convert to binary
    # cropped_img = crop_image(bw_img)
    # FeatureVector= getFeatureVector(cropped_img)
    # print(FeatureVector)

    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()