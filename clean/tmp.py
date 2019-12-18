# # ## SKEW DETECTION.
# # #1- Binarizing the image.
# # im_name = "capr1.png"
# # image = cv2.imread(im_name)
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # gray = cv2.bitwise_not(gray)
# # thresh = cv2.threshold(gray, 0, 255,
# # 	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# # cv2.imshow("thresh.png", thresh)

# # # Detecting the skew angle using column_stack.
# # coords = np.column_stack(np.where(thresh > 0))
# # angle = cv2.minAreaRect(coords)[-1]

# # if angle < -45:
# # 	angle = -(90 + angle)
# # else:
# # 	angle = -angle
# # #Rotating the image.
# # (h, w) = image.shape[:2]
# # center = (w // 2, h // 2)
# # M = cv2.getRotationMatrix2D(center, angle, 1.0)
# # rotated = cv2.warpAffine(image, M, (w, h),
# # 	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# # #cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
# # #	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# # #Removing noise. should be replaced by another technique in case of arabic.
# # #blur = cv2.GaussianBlur(rotated,(5,5),0)


# # # show the output image
# # print("[INFO] angle: {:.3f}".format(angle))
# # cv2.imwrite("Rotated.png",rotated)
# # rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
# # rotated_gray = cv2.bitwise_not(rotated_gray)


# #counting number of zeros in each row.
# # hist = np.count_nonzero(rotated_gray, axis=1)
# # i = 0
# # while(i<len(hist)): 
# #     count = 0
# #     if(hist[i] == 0):
# #         j = 0
# #         while(i+j <len(hist)):
# #             if(hist[i+j]==0):
# #                 j+=1
# #                 count +=1
# #             else:
# #                 break
# #         rotated_gray[i+int(count/2),:] = 255
# #         i+=j
# #     else:
# #         i+=1
# # cv2.imwrite("Rotated.png",rotated_gray)
# # line = 1
# # hist2 = np.count_nonzero(rotated_gray == 0 , axis=1)


# # bounds = []

# # for i in range(len(hist2)):
# #     if (hist2[i] == 0):
# #         bounds.append(i)
# # print(bounds)
# # for i in range(len(bounds)-1):
# #     result = rotated_gray[bounds[i]+1:bounds[i+1],:]
# #     cv2.imwrite("line"+str(line)+".png",result)
# #     line+=1

# ##############################################




# for i in range(line):
#     im_name = "line"+str(i+1)+".png"

#     image = cv2.imread(im_name)

#     hist3 = np.count_nonzero(image, axis=0)
#     k = 0
#     while(k<len(hist3)): 
#         count = 0
#         if(hist3[k].all() == 0):
#             j = 0
#             while(k+j <len(hist3)):
#                 if(hist3[k+j].all()==0):
#                     j+=1
#                     count +=1
#                 else:
#                     break
#             if(count>1):
#                 image[:,k+int(count/2)] = 255
#             k+=j
#         else:
#             k+=1

#     cv2.imwrite("line"+str(i+1)+".png",image)

### NOTE: character segmentation stuff.
# # Apply edge detection method on the image 
# edges = cv2.Canny(thresh,50,500,apertureSize = 3)
# #print(edges) 
# #print(np.count_nonzero(edges,axis=0))
# cv2.imwrite("edges.png",edges)

# # This returns an array of r and theta values 
# #lines = cv2.HoughLines(edges,0.0625,np.pi/180,threshold = 50)
# lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold = 50,minLineLength = 5 ,maxLineGap = 5 ) 
# print(lines)
# for i in range(len(lines)):
#     for x1,y1,x2,y2 in lines[i]:
#         cv2.line(edges,(x1,y1),(x2,y2),(255,0,0),2)
# cv2.imwrite(li)
# # The below for loop runs till r and theta values  
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
# cv2.imwrite('linesDetected.png', edges)