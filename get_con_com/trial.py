import cv2
import numpy as np





def resize_erode(img, scale, kernel_size ,iter):
    image = img.copy()
    (ys,xs) = img.shape
    image = cv2.resize(img,(xs*scale,ys*scale), interpolation=cv2.INTER_AREA)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    return cv2.erode(image,kernel,iterations = iter)

def resize_only(img, scale):
    image = img.copy()
    (ys,xs,_) = image.shape
    image = cv2.resize(image,(xs*scale,ys*scale), interpolation=cv2.INTER_AREA)
    return image 



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

def draw_vert_hist(bin_img, not_bin_img,threshold):
    im = bin_img.copy()
    im_2 = not_bin_img.copy()
    vert_hist = np.count_nonzero(im > 127, axis=0)
    k = 0
    bounds = []
    while(k<len(vert_hist)): 
        count = 1
        if(vert_hist[k] == 0):
            j = 1
            count = 1
            while(k+j <len(vert_hist)):
                if(vert_hist[k+j]== 0):
                    j+=1
                    count +=1
                else:
                    
                    break
            if(count > threshold):
                im[:,k+int(count/2)] = 255
                im_2[:,k+int(count/2)] = 0
                bounds.append(k+int(count/2))
            k+=j
        else:
            k+=1
    return im, im_2
    word = 1
    result_words = []
    for i in range(len(bounds)-1):
        result = im[:,bounds[i]+1:bounds[i+1]]
        result = im_2[:,bounds[i]+1:bounds[i+1]]
        result_words.append(result)
        word += 1
    return result_words_bin,result_words_not_bin




