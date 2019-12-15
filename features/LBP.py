import cv2
import numpy as np
from matplotlib import pyplot as plt 
from skimage.feature import local_binary_pattern

'''
boolean function used here f is simply
    = 1 if my neighbour is the same
    = 0 if it differs
'''
def LBP(img,r,b):  #r (radius) & b(# of neighbours) are  hyper parameters
    lbp = local_binary_pattern(img, b, r, method="default")  #{‘default’, ‘ror’, ‘uniform’,'nri_uniform', ‘var’}
    (hist, _) = np.histogram(lbp.ravel(),
        bins=np.arange(0, b + 3),
        range=(0, b + 2))

    # normalize the histogram
    # hist = hist.astype("float")
    # eps=1e-7  #honstly i don't know what is it XD
    # hist /= (hist.sum() + eps)

    return hist


def main():
    #read image and convert it to binary 
    img = cv2.imread("../tests/haa.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray) #find countours need character to be white and background black
    thresh = cv2.threshold(gray, 127,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    LBP_hist = LBP(thresh,3,8)
    plt.hist(LBP_hist,bins = [0,20,40,60,80,100]) 
    plt.title("histogram") 
    plt.show()
    print(LBP_hist)


main()
