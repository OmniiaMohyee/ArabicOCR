import cv2
import numpy as np


# 1.exact width and height
def crop_image(img):
    img_height, img_width = img.shape
    first_row = img_height
    first_col = img_width
    last_row = 0
    last_col = 0
    for i in range(img_height):
        for j in range(img_width):
            if img[i][j] == 0:
                first_row = min(first_row, i)
                first_col = min(first_col, j)
                last_row = max(last_row, i)
                last_col = max(last_col, j)
    cropped = img[first_row:last_row, first_col:last_col]
    # cv2.imshow("img",cropped)
    return cropped

def horizontal_transitions(img):
    horizontal =0
    height, width = img.shape
    for i in range(height):
        for j in range(1, width):
            if (img[i][j] != img[i][j-1]):
                horizontal +=1
    return horizontal
# 4. Vertical transitions
def vertical_transitions(img):
    vertical = 0
    height, width = img.shape
    for i in range(height):
        for j in range(1, width):
            if img[i][j] != img[i-1][j]:
                vertical +=1
    return vertical

def get_Regions(img):
    height,width = img.shape
    Region_1 = img[:height//2, :width//2]
    Region_2 = img[:height//2, width//2+1:]
    Region_3 = img[height//2+1:, :width//2]
    Region_4 = img[height//2+1:, width//2+1:]
    Regions = [Region_1, Region_2, Region_3, Region_4]
    # cv2.imshow("r1",Region_1)
    # cv2.imshow("r2",Region_2)
    # cv2.imshow("r3",Region_3)
    # cv2.imshow("r4",Region_4)
    return Regions

def getFeatureVector(cropped_img):
    FeatureVector = []
    height, width = cropped_img.shape
    FeatureVector.append(height/width) #1. height/width
    char_size = np.sum(cropped_img == 0)
    b = char_size
    w = np.sum(cropped_img == 255)
    FeatureVector.append(char_size) # add.1 char size/area #where the foregroung is the black
    FeatureVector.append(b/w) #2.black/white
    FeatureVector.append(horizontal_transitions(cropped_img)) # 3. horizontal transitions
    FeatureVector.append(vertical_transitions(cropped_img)) # 4. Vertical transitions
    regions = get_Regions(cropped_img)
    black = []
    white = []
    for region in regions:
        b = np.sum(region == 0)
        w = np.sum(region == 255)
        black.append(b)
        white.append(w)
    for i in range(4):
        FeatureVector.append(black[i]/white[i])
        FeatureVector.append(black[i]/char_size) #add.2 distribution features: for each quadrat -> Q/A
    #distribution features: for halves
    FeatureVector.append((black[0]+black[1])/char_size) # U/A
    FeatureVector.append((black[2]+black[3])/char_size) # Lo/A
    FeatureVector.append((black[0]+black[2])/char_size) # Le/A
    FeatureVector.append((black[1]+black[3])/char_size) # R/A
    for i in range(3):
        for j in range(i+1, 4):
            FeatureVector.append(black[i]/black[j])
    num_contours, (cx, cy), nu = contours_and_cetroid(cropped_img)
    FeatureVector.append(num_contours) #add. may be helpful
    FeatureVector.append(cx) #add.centroid may be helpful
    FeatureVector.append(cy)
    FeatureVector.append(nu) #add.3 moments

    return FeatureVector

def contours_and_cetroid(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00']) #get char centroid
        cy = int(M['m01']/M['m00'])
    else:
        print("most probalblt there is error in segmentation, this is an empty image")
        cx, cy = 0, 0
    return len(contours), (cx, cy), M['nu12']  # M['nu12'] normalized cetral moment

if __name__ == "__main__":
    #read image
    img = cv2.imread('../tests/gem.png')
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) #black char
    image_gray = cv2.bitwise_not(img_gray) #white char
    image_thresholded = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    cropped_img = crop_image(image_thresholded)
    FeatureVector= getFeatureVector(cropped_img)
    thresholded = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    _, markers = cv2.connectedComponents(thresholded)
    print(np.amax(markers))

    # print("num of features:", len(FeatureVector))
    # print(FeatureVector)
    # mask = (cropped_img != 0 ) & (cropped_img != 255 )
    # print(cropped_img[mask])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
