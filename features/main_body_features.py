import numpy as np
import cv2 
import random as rng
rng.seed(12345)

def find_biggest_contour(image):
   image = image.copy()
   contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

   contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
   biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

   mask = np.zeros(image.shape, np.uint8)
   cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
   return biggest_contour, mask

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
    return cropped_img

def crop(gray):
    edges = cv2.Canny(gray,50,200,apertureSize = 3)
    # contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    biggest_contour, _ =  find_biggest_contour(edges)

    # contours_poly = [None]*len(biggest_contour)
    # boundRect = [None]*len(biggest_contour)
    for i, c in enumerate(biggest_contour):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
          
    drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    
    # for i in range(len(contours)):
    #     color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    #     cv2.drawContours(drawing, contours_poly, i, color)
    #     cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
    #       (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    contours_poly = cv2.approxPolyDP(c, 3, True)
    boundRect = cv2.boundingRect(contours_poly[i])
    i = 3
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(drawing, contours_poly, i, color)
    cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
        (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    
    cv2.imshow('Contours', drawing)
    # cv2.imshow("img",cropped_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 


def main_body_feat(cropped_img):
    c_area = np.sum(cropped_img == 1)
    # for i in range (4):
    #     print('x')


def main():
    #read image and convert it to binary 
    img = cv2.imread("../tests/haa.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray) #find countours need character to be white and background black
    thresh = cv2.threshold(gray, 127,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cropped_img = crop_image(thresh)
    cropped_img =  crop(gray)
    main_body_feat(cropped_img)

main()