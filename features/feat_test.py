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
            if (bw_img[i][j]==0):
                first_row = min(first_row,i)
                first_col = min(first_col,j)
                last_row = max (last_row,i)
                last_col = max (last_col,j)
    cropped_img = bw_img[first_row:last_row,first_col:last_col]
    # cv2.imshow("img",cropped_img)
    return cropped_img

def horizontal_transitions(img):
    horizontal =0
    height, width = img.shape
    for i in range (height):
        for j in range(1,width):
            if (img[i][j] != img[i][j-1]):
                horizontal +=1
    return horizontal
# 4. Vertical transitions
def vertical_transitions(img):
    vertical = 0
    height, width = img.shape
    for i in range (height):
        for j in range (1,width):
            if (img[i][j] != img[i-1][j]):
                vertical +=1
    return vertical

def get_Regions(img):
    height,width = img.shape
    Region_1 = img[:height//2,:width//2]
    Region_2 = img[:height//2,width//2+1:]
    Region_3 = img[height//2+1:,:width//2]
    Region_4 = img[height//2+1:,width//2+1:]
    Regions = [Region_1,Region_2,Region_3,Region_4]
    return Regions

def getFeatureVector(cropped_img):
    # cv2.imwrite("cropped_img.png", cropped_img)
    FeatureVector=[]
    height, width = cropped_img.shape
    FeatureVector.append(height/width) #1. height/width
    # b,w = get_black_and_white(cropped_img)
    char_size = np.sum(cropped_img == 0)  # add.1 char size/area #where the foregroung is the black
    b = char_size
    w = np.sum(cropped_img == 1)
    FeatureVector.append(char_size) 
    FeatureVector.append(b/w) #2.black/white
    FeatureVector.append(horizontal_transitions(cropped_img)) # 3. horizontal transitions
    FeatureVector.append(vertical_transitions(cropped_img)) # 4. Vertical transitions

    regions = get_Regions(cropped_img)
    black=[]
    white=[]
    for i in range (len(regions)):
        # b,w = get_black_and_white(regions[i])
        b = np.sum(cropped_img == 0)
        w = np.sum(cropped_img == 1)
        black.append(b)
        white.append(w)
    for i in range (4):
        FeatureVector.append(black[i]/white[i])
        FeatureVector.append(black[i]/char_size) # distribution features: for each quadrat -> Q/A
    #distribution features: for halves
    FeatureVector.append((black[1]+black[2])/char_size) # U/A
    FeatureVector.append((black[3]+black[4])/char_size) # Lo/A
    FeatureVector.append((black[1]+black[3])/char_size) # Le/A
    FeatureVector.append((black[2]+black[4])/char_size) # R/A
    for i in range(3):
        for j in range(i+1,4):
            FeatureVector.append(black[i]/black[j])
    num_contours,(cx,cy) = contours_and_cetroid(bw_img)
    for 
    return FeatureVector

def contours_and_cetroid(thresh):
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00']) #get char centroid
    cy = int(M['m01']/M['m00'])
    return len(contours), (cx, cy)

if __name__=="__main__":
    #read image
    img = cv2.imread('../tests/t2214.png')
    img_grey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _, bw_img = cv2.threshold(img_grey,127,255,cv2.THRESH_BINARY) #convert to binary
    cropped_img = crop_image(bw_img)
    FeatureVector= getFeatureVector(cropped_img)
    print(FeatureVector)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()