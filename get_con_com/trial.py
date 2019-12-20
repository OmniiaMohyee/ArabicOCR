import cv2
import numpy as np





def resize_erode(img, scale, kernel_size ,iter):
    image = img.copy()
    (ys,xs) = img.shape
    image = cv2.resize(img,(xs*scale,ys*scale), interpolation=cv2.INTER_AREA)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    return cv2.erode(image,kernel,iterations = iter)



def imshow_components(img,name):
    
    ret, labels = cv2.connectedComponents(img)
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    #labeled_img = cv2.dilate(labeled_img, kernel,iterations = 1) 

    cv2.imwrite('word'+str(name)+'.png', labeled_img)
    return labels

def get_components(img,scale,kernel_size,iter,name):
    temp_img = resize_erode(img,scale,kernel_size,iter)
    return imshow_components(temp_img,name)





