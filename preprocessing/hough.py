import cv2 
import numpy as np 
  
# Reading the required image in  
# which operations are to be done.  
# Make sure that the image is in the same  
# directory in which this python program is 
img = cv2.imread('word_thresh') 
  
# Convert the img to grayscale 
im_name = "word.png"
image = cv2.imread(im_name)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("thresh.png", thresh)

  
# Apply edge detection method on the image 
edges = cv2.Canny(thresh,50,500,apertureSize = 3)
#print(edges) 
#print(np.count_nonzero(edges,axis=0))
cv2.imwrite("edges.png",edges)

# This returns an array of r and theta values 
#lines = cv2.HoughLines(edges,0.0625,np.pi/180,threshold = 50)
lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold = 50,minLineLength = 5 ,maxLineGap = 5 ) 
print(lines)
for i in range(len(lines)):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(edges,(x1,y1),(x2,y2),(255,0,0),2)
# The below for loop runs till r and theta values  
# are in the range of the 2d array 
# for i in range(0,len(lines)):
#     for r,theta in lines[i]: 
#         if(theta > np.pi/2 and theta < ((np.pi/2)+0.2)):
#     # Stores the value of cos(theta) in a 
#             print(lines[i])
#             a = np.cos(theta) 
  
#     # Stores the value of sin(theta) in b 
#             b = np.sin(theta) 
      
#     # x0 stores the value rcos(theta) 
#             x0 = a*r 
      
#     # y0 stores the value rsin(theta) 
#             y0 = b*r 
      
#     # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
#             x1 = int(x0 + 1000*(-b)) 
      
#     # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
#             y1 = int(y0 + 1000*(a)) 
  
#     # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
#             x2 = int(x0 - 1000*(-b)) 
      
#     # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
#             y2 = int(y0 - 1000*(a)) 
      
#     # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
#     # (0,0,255) denotes the colour of the line to be  
#     #drawn. In this case, it is red.  
#             cv2.line(edges,(x1,y1), (x2,y2), (255,0,0),2) 
      
# All the changes made in the input image are finally 
# written on a new image houghlines.jpg 
cv2.imwrite('linesDetected.png', edges)